# EAD_model

"""
This module contains the EAD default model
"""
import logging
import math

import numpy as np

from credit_studio_core.utils.cacheable import cacheable

LOGGER = logging.getLogger()


class EADModelDefault():
    """
    This is the default EAD (exposure at default) model used to return EAD amount for facilities. This class is
    subclassed by a model factory EAD class which decides, on a facility-by-facility basis, which ead model to use. The
    main public methods of this class are:

    get_performing_ead - the main method used for calculating EAD for a loan (EAD for delinquent loans is accounted for
    by changing the period in the call to this method - e.g. if a loan is 360 days past due the period passed into this
    function is the current calculation period minus 1 year)

    get_delinquent_cumulative_eads - the method used to calculate the full EAD array across all risk buckets. i.e.
    calculates EAD for all perfroming and non-performing risk buckets. This is needed for the stress testing code.

    """

    def __init__(self, data):
        self._period_length = 1  # [PYT-501]: could be changed to inherit period length from stress test
        self._data = data
        self._days_in_a_period = 360
        self._ead_model_default = EADModelDefault
        # [PYT-501]: the coll index should only be loaded in the macro data manager
        self._collateral_index = data.get_data_set('coll_dict')

    def get_delinquent_cumulative_eads(self, facility, calc_period, dpds=None, balance_evolution=False, scenario="None"):
        """
        Gets the cumulative delinquent Exposure-at-Default (EAD) for a single facility, for the single calculation
        period and an array of risk buckets. Performing risk buckets have the same EAD whereas the EAD for
        non-performing risk buckets is defined by the average days past due (dpd) for that risk bucket
        :param facility: a single loan facility or facility
        :param calc_period: period we want the EAD for
        :param numpy array dpds: array of dpd values to use in delinquent calculation (one for each risk bucket)
        :param balance_evolution: flag if this is being used for balance calculation
        :return: array of the delinquent EAD for each risk bucket
        """
        if dpds is None:
            dpds = [0]
        return np.vectorize(self._get_delinquent_cumulative_ead, otypes=['float'])(facility, calc_period, dpds)

    @cacheable(False, ['_full_key'], True, True)
    def _get_delinquent_cumulative_ead(self, facility, period_end, dpd):
        """
        Gets the cumulative, delinquent, Exposure-at-Default for a given facility-period combination.

        :param facility: a loan facility (or single loan)
        :param period_end: the period for which we want the cumulative delinquent EAD
        :param int dpd: days past due - the number of days a loan is overdue
        :return: delinquent_cumulative_ead: cumulative, delinquent EAD as a percentage for given facility-period
        """

        term = facility.get_term()

        # [PYT-501]: could add this to facility object
        starting_remaining_lifetime = (term -
                                       facility.get_time_since_origination(facility.get_start_date()) +
                                       self._period_length)

        pseudo_period_end = self._calc_psuedo_period_end(period_end, dpd, starting_remaining_lifetime)

        self._validate_dpd(facility, dpd, period_end)

        delinquent_cumulative_ead = self.get_performing_ead(facility, pseudo_period_end)

        return delinquent_cumulative_ead

    def get_performing_ead(self, facility, calc_period_end):
        """
        Returns the performing EAD for a given facility-period combination including bullet payments

        :param facility: a single loan or loan segment
        :param calc_period_end: period we want the EAD for

        :return: exposure_inc_bullet: the performing EAD for given facility-date combination
        """
        term = facility.get_term()
        rate = facility.get_relevant_eir(calc_period_end, self._data.get_data_set('vintage_buckets_eir'))
        facility_start_date = facility.get_start_date()
        partial_prepayment_rate = facility.get_partial_prepayment_rate()
        full_prepayment_rate = facility.get_full_prepayment_rate()
        total_prepayment_rate = partial_prepayment_rate + full_prepayment_rate
        bullet_share = facility.get_bullet_share()

        # [PYT-501]: add this to facility object
        starting_remaining_lifetime = (term -
                                       facility.get_time_since_origination(facility.get_start_date()) +
                                       self._period_length)

        bullet_amount_this_period = self._calc_bullet_payment_this_period(bullet_share, calc_period_end,
                                                                          starting_remaining_lifetime,
                                                                          facility_start_date)

        exposure_pre_bullet = self._calc_implied_exposure_pre_bullet(facility, calc_period_end,
                                                                     starting_remaining_lifetime, rate, bullet_share,
                                                                     total_prepayment_rate)

        exposure_inc_bullet = self._calc_exposure_inc_bullet(exposure_pre_bullet, bullet_amount_this_period)

        return exposure_inc_bullet

    # [PYT-501]: Move this onto the facility object or check that it doesn't already exist
    def _calc_remaining_lifetime(self, facility, period_end):
        """
        Calculate how many periods are left on a loans life at the BEGINNING of a period
        :param facility:
        :param period_end:
        :return:
        """
        term = facility.get_term()
        # [PYT-501]: Make this return remaining lifetime at end of period and all other calcs work off that
        remaining_lifetime = term - facility.get_time_since_origination(period_end) + 2 * self._period_length

        return remaining_lifetime

    def _validate_dpd(self, facility, dpd, period_end):
        """
        Validates whether the days past due makes sense for the given loan. Logic: If DPD greater than loan age in
        period log warning

        :param facility:
        :param dpd:
        :param period_end:
        :return:
        """

        starting_loan_age = facility.get_time_since_origination(facility.get_start_date()) - self._period_length

        if dpd > (starting_loan_age + period_end) * self._days_in_a_period:
            LOGGER.debug('Number of DPD in call to EAD _get_delinquent_cumulative_ead is larger '
                         'than the age of the current facility')
        elif dpd < 0:
            LOGGER.warning('DPD should be greater than 0')

        # [PYT-501]: possibly throw errors

    def _calc_psuedo_period_end(self, period_end, dpd, starting_remaining_lifetime):
        """
        Calculates a pesudo period end date based on the number of days a loan is past due. This effectively winds back
        the clock of the calculation period. Used to calculate exposure at time of default. An adjustment is made to
        ensure that if a loan is delinquent then the pseudo_period calculated is never greater than or equal to the loan
        lifetime. This ensures no bullet payment is added after a loan's lifetime if it is delinquent.

        :param period_end: calculation period end
        :param dpd: days past due
        :param starting_remaining_lifetime: remaining lifetime in start period of calculation
        :return:
        """
        # [PYT-501]: Make this work for non-year period lengths
        pseudo_period_end = period_end - dpd / self._days_in_a_period
        lifetime_periods_ceiling_value = 1000000

        # [PYT-501]: Make this adjustment cleaner
        if dpd > 0:
            # This ensures that if a loan is delinquent the pseudo_period_end is never greater than the lifetime of
            # a loan so that the bullet is never paid when a loan is delinquent
            post_lifetime_pseudo_period_end = starting_remaining_lifetime - 1 / self._days_in_a_period
        else:
            post_lifetime_pseudo_period_end = lifetime_periods_ceiling_value

        return np.min([pseudo_period_end, post_lifetime_pseudo_period_end])

    @staticmethod
    def _calc_bullet_payment_this_period(bullet_share, period_end, starting_remaining_lifetime, start_date):
        """
        Calculates the amount of the TOTAL bullet amount to be paid in the current period. If the loan is delinquent,
        i.e. dpd > 0 then no bullet is paid during the lifetime of the loan. If the calc_period is greater than the
        loan lifetime and dpd>0 this may return a bullet amount which is incorrect but this should not affect results.
        If dpd>0 really no bullet should be returned

        :param int bullet_share: the amount of the loan to be paid off as a final bullet (as a percentage of initial
                                exposure)
        :param period_end:
        :param starting_remaining_lifetime:
        :param start_date:
        :return:
        """
        return bullet_share * (period_end >= starting_remaining_lifetime + start_date)

    @staticmethod
    def _calc_exposure_inc_bullet(exposure_pre_bullet, bullet_amount_this_period):
        """
        Calculates the final EAD after contractual repayments and bullet
        :param exposure_pre_bullet:
        :param bullet_amount_this_period:
        :return:
        """
        return np.max([exposure_pre_bullet - bullet_amount_this_period, 0])

    def _calc_implied_exposure_pre_bullet(self, facility, pseudo_period_end,
                                          starting_remaining_lifetime, eir, bullet, prepayment_rate):
        """
        Calculates the exposure at default, pre-bullet payment including adjustments for laons that are delinquent.
        It interpolates between two integer periods.

        :param facility:
        :param pseudo_period_end:
        :param starting_remaining_lifetime:
        :param eir:
        :param bullet:
        :param prepayment_rate:

        :return:
        """

        previous_period_end = max(math.floor(pseudo_period_end), 0)
        # [PYT-501]: change this to handle non-integer period lengths
        prev_period_remaining_lifetime = self._calc_remaining_lifetime(facility, previous_period_end)

        next_period_end = max(math.ceil(pseudo_period_end), 0)
        # [PYT-501]: change this to handle non-integer period lengths
        next_period_remaining_lifetime = self._calc_remaining_lifetime(facility, next_period_end)

        facility_start_date = facility.get_start_date()

        prev_exp = self._calc_exposure_pre_bullet(facility_start_date, previous_period_end,
                                                  prev_period_remaining_lifetime, starting_remaining_lifetime,
                                                  eir, bullet, prepayment_rate)
        next_exp = self._calc_exposure_pre_bullet(facility_start_date, next_period_end, next_period_remaining_lifetime,
                                                  starting_remaining_lifetime, eir, bullet, prepayment_rate)

        return np.max([0, next_exp + (next_period_end - pseudo_period_end) * (prev_exp - next_exp)])

    def _calc_exposure_pre_bullet(self, facility_start_date, period_end, remaining_lifetime,
                                  starting_remaining_lifetime, eir, bullet_share, prepayment_rate):
        """
        Calculate the exposure pre bullet payment for a given facility and calculation period

        :param facility_start_date:
        :param period_end:
        :param remaining_lifetime:
        :param starting_remaining_lifetime:
        :param eir:
        :param bullet_share:
        :param prepayment_rate:
        :return:
        """
        if facility_start_date >= period_end:
            # Return 1 if the calculation period is pre the existence of the loan
            return 1

        if period_end >= starting_remaining_lifetime:
            # If period end > number of periods left on loan on calculation start date return the bullet
            return bullet_share

        exposure = self._calc_exposure_pre_bullet(facility_start_date, period_end - 1, remaining_lifetime + 1,
                                                  starting_remaining_lifetime, eir, bullet_share, prepayment_rate)

        prev_period_exp_inc_prepay = exposure

        # Add interest
        exposure += exposure * eir
        # Remove payments
        exposure += np.pmt(eir, np.max([remaining_lifetime, 1]), prev_period_exp_inc_prepay, -bullet_share)
        # Remove Prepayments
        exposure -= exposure * prepayment_rate

        return exposure

    @cacheable(False, ['_full_key'], True, False)
    def get_adjusted_ltv(self, facility, period, config):
        """
        Evolve the LTV forwarded one period by calculating change in collateral value and change in EAD.

       ---Calculation---

        LTV(t+1)_i = LTV(t)_i * (CollateralValue(t-1)/CollateralValue(t)) * (EAD(t)_i/EAD(t-1)_i)

        ---Explanation of <EAD(t)>---
        The ith component of the EAD vector, <EAD(t)>_i, contains the proportion of the TOTAL initial (initial as in
        origination or reporting?) exposure that remains, given that the loan is in risk bucket i at t. The EAD
        should not be confused with the Exposure. Exposure(t)_i is the value of the facility in risk bucket i at time t,
        the TOTAL exposure should be constant in time. In contrast, the EAD is not a distribution and therefore its
        sum is not a meaningful quantity (and therefore does not need to stay constant).

        The remaining exposure should be the same for all PE buckets, since if the loan is in a PE bucket it will
        have (by definition) made all of its required payments. The remaining exposure in NPE buckets will depend on
        the average Days Past Due of a loan in that bucket, the higher the average DPD, the higher the remaining
        exposure.

        ---Explanation of CollateralValue(t)---
        CollateralValue(t) is a scalar. The value of the collateral at the reporting date changes with macroeconomic
        conditions.

        :param facility:
        :param period:
        :return: array of LTVs,  <LTV(t)> = <EAD(t)> / CollateralValue(t), where <EAD(t)> is a vector.
        """
        facility_start_date = facility.result_storage.get_start_date()
        average_dpds = self._get_average_dpds(facility)
        enable_ltv_adjustment = config['Runtime Parameters'].getboolean('Enable_LTV_Adjustment')

        if period <= facility_start_date + self._period_length:
            comparison_period = facility_start_date
            reporting_ltv = facility.get_ltv()
            ltvs = np.repeat(reporting_ltv, len(average_dpds))
        else:
            comparison_period = period - self._period_length
            ltvs = self.get_adjusted_ltv(facility, comparison_period, config)

        if enable_ltv_adjustment:
            new_ltvs = ltvs * self._get_collateral_value_multiplier(facility, period, comparison_period)
            new_ltvs = self._ead_adjustment(new_ltvs, period, comparison_period, facility, average_dpds)
            return self._round_off_ltv(new_ltvs)
        return self._round_off_ltv(ltvs)

    def _ead_adjustment(self, ltvs, period, comparison_period, facility, average_dpds):
        """
        Element-wise adjust LTV due to change in ead (ie, remaining loan balance).

        Where ead now is 0, do not divide: in this case, we want an LTV 0 anyway and the division may result in a
        divide by zero error.
        """
        eads_now = self._get_cumulative_eads(facility, period, average_dpds)
        eads_at_comparison = self._get_cumulative_eads(facility, comparison_period, average_dpds)
        return np.divide(ltvs * eads_now, eads_at_comparison, where=eads_now != 0)

    def _get_collateral_value_multiplier(self, facility, base_period, comparison_period):
        return self._get_collateral_index(facility, comparison_period) / self._get_collateral_index(facility,
                                                                                                    base_period)

    def _get_collateral_index(self, facility, period):
        # [PYT-501]: add migration matrix model as attribute of ead model, and commonise this function
        return self.get_relevant_collateral_index_value(facility, period)

    def _get_average_dpds(self, facility):
        # [PYT-501]: add migration matrix model as attribute of ead model, and commonise this function
        bucket_dataset = self._data.get_data_set('rating_grades_data')
        avg_dpds = np.array(bucket_dataset['average_dpd'])
        avg_dpds = np.append(avg_dpds, [0])
        return avg_dpds

    def _get_cumulative_eads(self, facility, period, dpds):
        return self.get_delinquent_cumulative_eads(facility, period, dpds)

    def _get_amortization_buckets(self, facility):
        # [PYT-501]: add migration matrix model as attribute of ead model, and commonise this function
        bucket_dataset = self._data.get_data_set('rating_grades_data')
        amort_bckts = np.array(bucket_dataset['amortisation_bucket'])
        amort_bckts = np.append(amort_bckts, [1])
        return amort_bckts

    @staticmethod
    def _round_off_ltv(ltv):
        """# Round off ltv to 1%, with a floor to avoid division by zero."""
        ltv = np.round(ltv, 2)

        return np.maximum(ltv, 0.0025)

    def get_relevant_collateral_index_value(self, segment, period):
        # [PYT-501]: add macro data manager as attribute of ead model, and commonise this function
        """
        Extract the relevant collateral index value for a specific loan / loan segment and period

        :param segment: loan segment object
        :param period: the period to get collateral index for

        :return: int collateral index value for given loan segment and period
        """

        # retrieve relevant sub-segment of the collateral indexes
        relevant_coll_indexes = self.get_linked_collateral_index(segment)

        if period in relevant_coll_indexes:
            return relevant_coll_indexes[period]
        closest_period = min(relevant_coll_indexes.keys(), key=lambda x: abs(x - period))
        return relevant_coll_indexes[closest_period]

    def get_linked_collateral_index(self, segment):
        # [PYT-501]: add macro data manager as attribute of ead model, and commonise this function
        """
        Returns the relevant collateral indexes for a given loan segment

        :param segment: loan segment object
        :return: array relevant_coll_indexes: a list of the collateral indexes linked to the given loan segment
        """
        # retrieve collateral index key which is stored to each segment
        coll_index_key = segment.get_coll_index_key()
        # retrieve relevant sub-segment of the collateral index
        relevant_coll_indexes = self._collateral_index[coll_index_key]
        return relevant_coll_indexes
