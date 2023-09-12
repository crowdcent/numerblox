import os
import time
import warnings
import numpy as np
import pandas as pd
import datetime as dt
import pandas_ta as ta
from tqdm.auto import tqdm
from functools import wraps
from scipy.stats import rankdata
from abc import ABC, abstractmethod
from rich import print as rich_print
from typing import Union, Tuple, List
from multiprocessing.pool import Pool
from sklearn.linear_model import Ridge
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import QuantileTransformer

from .numerframe import NumerFrame

class BaseProcessor(ABC):
    """Common functionality for preprocessors and postprocessors."""

    def __init__(self):
        ...

    @abstractmethod
    def transform(
        self, dataf: Union[pd.DataFrame, NumerFrame], *args, **kwargs
    ) -> NumerFrame:
        ...

    def __call__(
        self, dataf: Union[pd.DataFrame, NumerFrame], *args, **kwargs
    ) -> NumerFrame:
        return self.transform(dataf=dataf, *args, **kwargs)


def display_processor_info(func):
    """Fancy console output for data processing."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = dt.datetime.now()
        result = func(*args, **kwargs)
        time_taken = str(dt.datetime.now() - tic)
        class_name = func.__qualname__.split(".")[0]
        rich_print(
            f":white_check_mark: Finished step [bold]{class_name}[/bold]. Output shape={result.shape}. Time taken for step: [blue]{time_taken}[/blue]. :white_check_mark:"
        )
        return result

    return wrapper


class CopyPreProcessor(BaseProcessor):
    """Copy DataFrame to avoid manipulation of original DataFrame."""

    def __init__(self):
        super().__init__()

    @display_processor_info
    def transform(self, dataf: Union[pd.DataFrame, NumerFrame]) -> NumerFrame:
        return NumerFrame(dataf.copy())


class FeatureSelectionPreProcessor(BaseProcessor):
    """
    Keep only features given + all target, predictions and aux columns.
    """

    def __init__(self, feature_cols: Union[str, list]):
        super().__init__()
        self.feature_cols = feature_cols

    @display_processor_info
    def transform(self, dataf: NumerFrame) -> NumerFrame:
        keep_cols = (
            self.feature_cols
            + dataf.target_cols
            + dataf.prediction_cols
            + dataf.aux_cols
        )
        dataf = dataf.loc[:, keep_cols]
        return NumerFrame(dataf)


class TargetSelectionPreProcessor(BaseProcessor):
    """
    Keep only features given + all target, predictions and aux columns.
    """

    def __init__(self, target_cols: Union[str, list]):
        super().__init__()
        self.target_cols = target_cols

    @display_processor_info
    def transform(self, dataf: NumerFrame) -> NumerFrame:
        keep_cols = (
            self.target_cols
            + dataf.feature_cols
            + dataf.prediction_cols
            + dataf.aux_cols
        )
        dataf = dataf.loc[:, keep_cols]
        return NumerFrame(dataf)


class ReduceMemoryProcessor(BaseProcessor):
    """
    Reduce memory usage as much as possible.

    Credits to kainsama and others for writing about memory usage reduction for Numerai data:
    https://forum.numer.ai/t/reducing-memory/313

    :param deep_mem_inspect: Introspect the data deeply by interrogating object dtypes.
    Yields a more accurate representation of memory usage if you have complex object columns.
    """

    def __init__(self, deep_mem_inspect=False):
        super().__init__()
        self.deep_mem_inspect = deep_mem_inspect

    @display_processor_info
    def transform(self, dataf: Union[pd.DataFrame, NumerFrame]) -> NumerFrame:
        dataf = self._reduce_mem_usage(dataf)
        return NumerFrame(dataf)

    def _reduce_mem_usage(self, dataf: pd.DataFrame) -> pd.DataFrame:
        """
        Iterate through all columns and modify the numeric column types
        to reduce memory usage.
        """
        start_memory_usage = (
            dataf.memory_usage(deep=self.deep_mem_inspect).sum() / 1024**2
        )
        rich_print(
            f"Memory usage of DataFrame is [bold]{round(start_memory_usage, 2)} MB[/bold]"
        )

        for col in dataf.columns:
            col_type = dataf[col].dtype.name

            if col_type not in [
                "object",
                "category",
                "datetime64[ns, UTC]",
                "datetime64[ns]",
            ]:
                c_min = dataf[col].min()
                c_max = dataf[col].max()
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        dataf[col] = dataf[col].astype(np.int16)
                    elif (
                        c_min > np.iinfo(np.int16).min
                        and c_max < np.iinfo(np.int16).max
                    ):
                        dataf[col] = dataf[col].astype(np.int16)
                    elif (
                        c_min > np.iinfo(np.int32).min
                        and c_max < np.iinfo(np.int32).max
                    ):
                        dataf[col] = dataf[col].astype(np.int32)
                    elif (
                        c_min > np.iinfo(np.int64).min
                        and c_max < np.iinfo(np.int64).max
                    ):
                        dataf[col] = dataf[col].astype(np.int64)
                else:
                    if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                    ):
                        dataf[col] = dataf[col].astype(np.float16)
                    elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                    ):
                        dataf[col] = dataf[col].astype(np.float32)
                    else:
                        dataf[col] = dataf[col].astype(np.float64)

        end_memory_usage = (
            dataf.memory_usage(deep=self.deep_mem_inspect).sum() / 1024**2
        )
        rich_print(
            f"Memory usage after optimization is: [bold]{round(end_memory_usage, 2)} MB[/bold]"
        )
        rich_print(
            f"[green] Usage decreased by [bold]{round(100 * (start_memory_usage - end_memory_usage) / start_memory_usage, 2)}%[/bold][/green]"
        )
        return dataf


class BayesianGMMTargetProcessor(BaseProcessor):
    """
    Generate synthetic (fake) target using a Bayesian Gaussian Mixture model. \n
    Based on Michael Oliver's GitHub Gist implementation: \n
    https://gist.github.com/the-moliver/dcdd2862dc2c78dda600f1b449071c93

    :param target_col: Column from which to create fake target. \n
    :param feature_names: Selection of features used for Bayesian GMM. All features by default.
    :param n_components: Number of components for fitting Bayesian Gaussian Mixture Model.
    """

    def __init__(
        self,
        target_col: str = "target",
        feature_names: list = None,
        n_components: int = 6,
    ):
        super().__init__()
        self.target_col = target_col
        self.feature_names = feature_names
        self.n_components = n_components
        self.ridge = Ridge(fit_intercept=False)
        self.bins = [0, 0.05, 0.25, 0.75, 0.95, 1]

    @display_processor_info
    def transform(self, dataf: NumerFrame, *args, **kwargs) -> NumerFrame:
        all_eras = dataf[dataf.meta.era_col].unique()
        coefs = self._get_coefs(dataf=dataf, all_eras=all_eras)
        bgmm = self._fit_bgmm(coefs=coefs)
        fake_target = self._generate_target(dataf=dataf, bgmm=bgmm, all_eras=all_eras)
        dataf[f"{self.target_col}_fake"] = fake_target
        return NumerFrame(dataf)

    def _get_coefs(self, dataf: NumerFrame, all_eras: list) -> np.ndarray:
        """
        Generate coefficients for BGMM.
        Data should already be scaled between 0 and 1
        (Already done with Numerai Classic data)
        """
        coefs = []
        for era in all_eras:
            features, target = self.__get_features_target(dataf=dataf, era=era)
            self.ridge.fit(features, target)
            coefs.append(self.ridge.coef_)
        stacked_coefs = np.vstack(coefs)
        return stacked_coefs

    def _fit_bgmm(self, coefs: np.ndarray) -> BayesianGaussianMixture:
        """
        Fit Bayesian Gaussian Mixture model on coefficients and normalize.
        """
        bgmm = BayesianGaussianMixture(n_components=self.n_components)
        bgmm.fit(coefs)
        # make probability of sampling each component equal to better balance rare regimes
        bgmm.weights_[:] = 1 / self.n_components
        return bgmm

    def _generate_target(
        self, dataf: NumerFrame, bgmm: BayesianGaussianMixture, all_eras: list
    ) -> np.ndarray:
        """Generate fake target using Bayesian Gaussian Mixture model."""
        fake_target = []
        for era in tqdm(all_eras, desc="Generating fake target"):
            features, _ = self.__get_features_target(dataf=dataf, era=era)
            # Sample a set of weights from GMM
            beta, _ = bgmm.sample(1)
            # Create fake continuous target
            fake_targ = features @ beta[0]
            # Bin fake target like real target
            fake_targ = (rankdata(fake_targ) - 0.5) / len(fake_targ)
            fake_targ = (np.digitize(fake_targ, self.bins) - 1) / 4
            fake_target.append(fake_targ)
        return np.concatenate(fake_target)

    def __get_features_target(self, dataf: NumerFrame, era) -> tuple:
        """Get features and target for one era and center data."""
        sub_df = dataf[dataf[dataf.meta.era_col] == era]
        features = self.feature_names if self.feature_names else sub_df.feature_cols
        target = sub_df[self.target_col].values - 0.5
        features = sub_df[features].values - 0.5
        return features, target


V4_2_FEATURE_GROUP_MAPPING = {"intelligence": ["feature_ethiopic_anhedonic_stob", "feature_pulverized_unified_dupery", "feature_pansophical_agitato_theatricality", "feature_recreational_homiletic_nubian", "feature_received_veiniest_tamarix", "feature_paleolithic_myalgic_lech", "feature_phasmid_accommodating_aftercare", "feature_untinned_dextrogyrate_fining", "feature_unsatisfactory_lovelorn_brainstorm", "feature_terminative_extroverted_interpretation", "feature_pacific_sleeved_devise", "feature_unconstitutional_chiseled_coalport", "feature_superhuman_coenobitical_scotticism", "feature_wendish_synchronal_advertency", "feature_unventilated_sollar_bason", "feature_godliest_consistorian_woodpecker", "feature_weedier_indistinctive_haranguer", "feature_faddiest_clement_repatriation", "feature_monocyclic_galliambic_par", "feature_protonematal_springtime_varioloid", "feature_instructional_desensitized_symmetallism", "feature_disarrayed_rarefactive_trisulphide", "feature_antarthritic_syzygial_wonderland", "feature_guardant_giocoso_natterjack", "feature_ten_male_phoenician", "feature_guardant_irrepealable_onanist", "feature_promised_intramolecular_sora", "feature_bespoke_combining_farrow", "feature_unliquefied_folding_equilibrator", "feature_uncomplying_unprovocative_ochlophobia", "feature_differing_lenticular_gossiping", "feature_ottoman_metaphysical_parathion", "feature_variolate_reducible_sweet", "feature_jumpiest_clattering_pentstemon", "feature_sylphid_maladaptive_franc"], "charisma": ["feature_supercelestial_telic_dyfed", "feature_altimetrical_muddled_symbolism", "feature_unshaken_ahorse_wehrmacht", "feature_gambrel_unblessed_gigantomachy", "feature_obeliscal_bewildered_reviewer", "feature_underdressed_tanagrine_prying", "feature_corniest_undue_scall", "feature_reduplicative_appalling_metastable", "feature_wrathful_prolix_colotomy", "feature_limonitic_issuable_melancholy", "feature_approximal_telautographic_sharkskin", "feature_fribble_gusseted_stickjaw", "feature_spec_subversive_plotter", "feature_unsinkable_dumbstruck_octuplet", "feature_integrative_reviviscent_governed", "feature_tamil_grungy_empathy", "feature_canopic_exigible_schoolgirl", "feature_plumular_constantinian_repositing", "feature_serpentiform_trinary_imponderability", "feature_gyroidal_embowed_pilcher", "feature_unlivable_armenian_wedge", "feature_flawed_demonological_toady", "feature_pruinose_raploch_roubaix", "feature_seediest_ramshackle_reclamation", "feature_hagiological_refer_vitamin", "feature_alcibiadean_lumpier_origan", "feature_encased_unamiable_hasidism", "feature_evocable_woollen_guarder", "feature_hunchbacked_unturning_meditation", "feature_circumnavigable_naughty_retranslation", "feature_testicular_slashed_ventosity", "feature_potential_subsessile_disconnection", "feature_unswaddled_inenarrable_goody", "feature_stellular_paler_centralisation", "feature_angevin_fitful_sultan", "feature_subinfeudatory_brainy_carmel", "feature_simpatico_cadential_pup", "feature_esculent_erotic_epoxy", "feature_milliary_hyperpyretic_medea", "feature_coraciiform_sciurine_reef", "feature_weightiest_protozoic_brawler", "feature_certificated_putrescent_godship", "feature_deckled_exaggerative_algol", "feature_unrecognisable_waxier_paging", "feature_fortyish_neptunian_catechumenate", "feature_tricksiest_pending_voile", "feature_forcipate_laced_greenlet", "feature_scombroid_genoese_kilojoule", "feature_topologic_subcaliber_radiotelephony", "feature_diluted_maxillary_denouement", "feature_flauntier_forethoughtful_festival", "feature_emarginate_enigmatical_yiddish", "feature_planular_naturistic_pinhead", "feature_insipid_unshakable_panne", "feature_abreast_viscoelastic_commander", "feature_uneffaced_unintermitted_spurt", "feature_tenth_contrasting_slice", "feature_geomorphological_uncompanioned_colander", "feature_deflective_demographic_saragossa", "feature_teratogenic_wet_calkin", "feature_graceful_vaunted_accommodator", "feature_perpetuable_stuffed_manxman", "feature_belittled_scenographic_antidisestablishmentarianism", "feature_plausive_skaldic_shoebox", "feature_incognita_cleaned_asphyxiant", "feature_biaxal_forworn_sixty", "feature_dressy_skiable_hypognathism", "feature_maltese_flaggiest_platelayer", "feature_gory_medullated_waverer", "feature_precursory_maltese_wetting", "feature_overexcitable_filmier_queening", "feature_analgesic_ardent_conservatoire", "feature_realisable_defeatist_japer", "feature_adventive_isohyetal_newshawk", "feature_interatomic_doggish_teutonisation", "feature_peeled_singing_smacking", "feature_sophomoric_perseverant_sclaff", "feature_regrettable_liberating_crabber", "feature_polyzoarial_histologic_shallop", "feature_revitalizing_intersectional_dysthymia", "feature_sanctioned_sunny_lily", "feature_chariest_unimplored_towboat", "feature_unchaste_raisable_tetraploidy", "feature_disapproving_behind_dampener", "feature_instinctive_dappled_upholder", "feature_unforeboding_succeeding_wildcatter", "feature_cockiest_ditheistical_pennyworth", "feature_baggier_supernormal_calcedony", "feature_convergent_reborn_autoroute", "feature_choric_illustrated_inch", "feature_uninquiring_unspiritualized_gradualism", "feature_embolic_menial_mariculture", "feature_pectoral_couthie_epiphenomenon", "feature_literal_memoriter_heavy", "feature_fatalistic_cramoisy_locative", "feature_grassier_sizy_chorale", "feature_quaternate_loftier_directorate", "feature_stunning_unladen_ockham", "feature_prototypical_daylong_glop", "feature_leafiest_unrepaired_indemnification", "feature_ecumenical_neuron_equity", "feature_castrated_inculpatory_pea", "feature_unstarched_isogonic_alternation", "feature_unthawed_moved_theft", "feature_depleted_omnidirectional_baluchi", "feature_unsocial_pouring_barbe", "feature_like_inflorescent_sierra", "feature_mauve_supernumerary_hogan", "feature_harmonized_intrinsic_disapproval", "feature_unplumb_prefectorial_gloucester", "feature_towery_eighteenth_enough", "feature_kinematical_absorbable_specialisation", "feature_lamplit_unengaged_mommy", "feature_unmoaned_attritional_crenation", "feature_deltoid_chalkier_connecticut", "feature_disincentive_unchallenged_aerobe", "feature_regurgitate_demolition_downstate", "feature_antenuptial_moonlit_incidence", "feature_disclosed_transcendental_pungency", "feature_certified_sensorial_occiput", "feature_froward_flattering_cretic", "feature_packaged_topological_paradigm", "feature_interfemoral_battered_ghetto", "feature_intrusive_sniffy_gangland", "feature_pedicular_fanfold_beseecher", "feature_effortful_inenarrable_johnsonianism", "feature_tardigrade_intercommunal_propitiatory", "feature_elasmobranch_braving_typhoid", "feature_aweary_fruiting_climb", "feature_unwomanly_pitying_saffian", "feature_aeriform_discomposed_moolvie", "feature_laniary_chelated_rejoicing", "feature_tricksome_unimproved_accidental", "feature_unperched_forgetive_vasoconstrictor", "feature_cislunar_ritardando_gadwall", "feature_odd_desiderative_godet", "feature_natant_wriest_dollop", "feature_tiptoe_decadent_statue", "feature_quaggy_chilliest_inchoative", "feature_explanatory_immature_cautery", "feature_versicular_buoyant_codetta", "feature_septuple_bonapartean_sanbenito", "feature_probationary_readying_roundelay", "feature_manifold_melodramatic_girl", "feature_sardonic_primary_shadwell", "feature_snatchy_xylic_institution", "feature_multiseriate_oak_benzidine", "feature_gobioid_transhuman_interconnection", "feature_reservable_peristomal_emden", "feature_inestimable_unmoral_extraversion", "feature_nubby_sissified_value", "feature_incorporating_abominable_daily", "feature_herbaged_brownish_consubstantialist", "feature_solemn_wordier_needlework", "feature_evangelistic_cruel_dissimilitude", "feature_impetratory_shuttered_chewer", "feature_referenced_biliteral_chiropody", "feature_eleatic_fellow_auctioneer", "feature_malpighian_vaporized_biogen", "feature_expiscatory_wriest_colportage", "feature_yelled_hysteretic_eath", "feature_bitterish_buttocked_turtleneck", "feature_percipient_atelectatic_cinnamon", "feature_gobony_premonitory_twinkler", "feature_twittery_tai_attainment", "feature_crooked_wally_lobation", "feature_crookback_workable_infringement", "feature_brawling_unpeppered_comedian", "feature_glyphographic_reparable_empyrean", "feature_noctilucent_subcortical_proportionality", "feature_guardian_frore_rolling", "feature_denuded_typed_wattmeter", "feature_unreachable_neritic_saracen", "feature_enzymatic_poorest_advocaat", "feature_wariest_vulnerable_unmorality", "feature_guttering_half_spondee", "feature_distressed_bloated_disquietude", "feature_leaky_overloaded_rhodium", "feature_unsapped_anionic_catherine", "feature_kissable_forfeit_egotism", "feature_unsizable_ancestral_collocutor", "feature_healthier_unconnected_clave", "feature_unimproved_courtliest_uncongeniality", "feature_coordinated_astir_vituperation", "feature_coactive_bandoleered_trogon", "feature_bootleg_clement_joe", "feature_thousandth_hierarchal_plight", "feature_unhoped_hex_ventriloquism", "feature_treacly_heuristic_binding", "feature_bulbed_viscose_toy", "feature_patristic_intelligential_crackajack", "feature_lucullian_unshunned_ulex", "feature_unrealistic_inoperable_publishing", "feature_perceptive_unnavigable_elegancy", "feature_recurrent_perversive_injury", "feature_overrank_shavian_epitaxy", "feature_petticoated_unobeyed_mugwort", "feature_stimulant_proximal_moocher", "feature_datival_crucial_chevrotain", "feature_just_flavescent_draff", "feature_cephalopod_arrased_jird", "feature_endogamic_suasible_boasting", "feature_rheumatic_bravest_pantisocracy", "feature_store_comforted_goiter", "feature_goofy_glassed_cetacea", "feature_crushed_gleg_reintroduction", "feature_scald_vanishing_enchainment", "feature_blinded_memorable_wring", "feature_clavate_geriatric_aldebaran", "feature_palimpsest_inoffensive_coiffeuse", "feature_adored_empyreal_revel", "feature_schizomycetic_gooey_mynah", "feature_pourable_multangular_requisition", "feature_resuscitated_taloned_borak", "feature_devoured_disunited_didn't", "feature_undefeated_unworkmanlike_probe", "feature_archaean_unregarded_caravel", "feature_concealed_artful_thaw", "feature_bluff_carbonyl_verbena", "feature_gymnorhinal_unreached_carthusian", "feature_falsifiable_performative_maxixe", "feature_vapourish_ichthyotic_causerie", "feature_craggier_windier_apologia", "feature_elaborate_intimate_bor", "feature_meteorological_tushed_ester", "feature_iffy_pretty_gumming", "feature_numeral_cagey_haulm", "feature_piping_geotactic_cusp", "feature_rutilant_obliterate_potassium", "feature_unhomely_monoclinic_acajou", "feature_adynamic_ramstam_urtica", "feature_khedival_viewable_bloodlust", "feature_petulant_thirty_safety", "feature_manganic_hyetographical_cadastre", "feature_quinoidal_ungrudged_hebraiser", "feature_intersectional_inseminated_undersigned", "feature_idyllic_spectacled_togue", "feature_antiskid_circumlocutional_ogdon", "feature_overdone_raspiest_halcyon", "feature_coagulatory_bathymetrical_pergamum", "feature_paragenetic_traditional_madeline", "feature_collapsable_sinful_cassareep", "feature_preceding_perturbing_radii", "feature_hypogeal_unburied_embraceor", "feature_hobbesian_infrasonic_subjugator", "feature_deliquescent_pelvic_functionary", "feature_maternal_knobbier_dialecticism", "feature_blowzier_sleepiest_verifier", "feature_uncurtailed_sabaean_ode", "feature_crosscut_rompish_osteopathist", "feature_expectative_intimidated_bluffer", "feature_creaking_unsophisticated_clavier", "feature_prissy_counsellable_keg", "feature_unreposing_mellifluent_dindle", "feature_cowled_knottiest_megilp", "feature_yoruban_unapplied_tawse", "feature_electrophysiological_blearier_deconsecration", "feature_nepotic_overreaching_itinerary", "feature_scrawny_wavy_pheon", "feature_incurrent_homeopathic_overcapacity", "feature_viridescent_choking_pinto", "feature_julienne_interludial_noradrenaline", "feature_unexplained_orthorhombic_kenny", "feature_aspheric_cloudy_restorationism", "feature_epistatic_absent_slayer", "feature_won_stalwart_eisenstein", "feature_indulgent_amphibrachic_surrebutter", "feature_hypnagogic_wedded_religionist", "feature_dormant_samariform_elucidator", "feature_geostationary_manky_teutonisation", "feature_unabbreviated_unteamed_krasnoyarsk", "feature_deontological_sidearm_alder", "feature_saprogenic_quadrilateral_chook", "feature_hagiographical_fertile_disestablishment", "feature_moonlit_hundred_conferrer", "feature_immensurable_myrmecological_angler", "feature_telencephalic_assumed_loran", "feature_other_peatier_raymond", "feature_unkingly_protochordate_confluent", "feature_falcate_concurrent_kwa", "feature_snuggest_counterclockwise_desolater", "feature_perspectivist_nondestructive_haemocoel", "feature_demonstrated_wordiest_globulin", "feature_wrinkliest_unmaintainable_usk", "feature_unhurt_centroidal_whimsey", "feature_supratemporal_pharmaceutical_ambassadorship", "feature_unordered_insertional_looter", "feature_geothermal_optional_octagon", "feature_overnight_fluent_trepan", "feature_necrotic_potamic_passionary"], "strength": ["feature_tuberculate_patelliform_paging", "feature_inflammable_numb_anticline", "feature_malignant_campodeid_pluton", "feature_dure_jaspery_mugging", "feature_educational_caustic_mythologisation", "feature_diverted_astral_dunghill", "feature_degenerate_diaphragmatic_literalizer", "feature_laced_scraggly_grimalkin", "feature_wheezier_unjaundiced_game", "feature_unimpressed_uninflected_theophylline", "feature_shiite_overfed_mense", "feature_irritant_reciprocal_pelage", "feature_bricky_runed_bottleful", "feature_phyletic_separate_genuflexion", "feature_peckish_impetrative_kanpur", "feature_unshrinking_semiarid_floccule", "feature_heartier_salverform_nephew", "feature_geostrophic_adaptative_karla", "feature_navigational_enured_condensability", "feature_confusable_pursy_plosion", "feature_clenched_wayward_coelostat", "feature_developed_arbitrary_traditionalist", "feature_unnameable_abysmal_net", "feature_completive_pedantical_sinecurist", "feature_witchy_orange_muley", "feature_misfeatured_sometime_tunneler", "feature_overjoyed_undriven_sauna", "feature_antacid_thermogenic_wilhelm", "feature_gerundival_puristic_gobioid", "feature_ministrative_slurred_parenchyma", "feature_squabbiest_receivable_foreyard", "feature_daytime_arching_expropriator", "feature_underdeveloped_eastern_goner", "feature_noetic_thermometric_pique", "feature_glial_birefringent_popery", "feature_rural_apogean_titbit", "feature_aaronic_unexampled_arguer", "feature_chorial_sapheaded_humberside", "feature_juvenalian_typical_langton", "feature_heterotactic_masculine_liquidity", "feature_piping_unabolished_statocyst", "feature_scroddled_groaning_sanctuary", "feature_overlarge_societal_alternator", "feature_unquestioning_dappled_phenomenalism", "feature_pacifist_unmeaning_haslet", "feature_cantharidian_lightful_cubicle", "feature_supportive_sublime_phenolic", "feature_derogate_bigoted_mnemonic", "feature_balinese_uncomforted_mimicker", "feature_revolutionist_proportionate_headscarf", "feature_anticivic_blistery_knot", "feature_anamnestic_conformable_spaceship", "feature_leprose_corroborant_incapacitation", "feature_sunset_slouchy_alisma", "feature_cased_polycrystalline_groschen", "feature_fremd_cricoid_gibberellin", "feature_knowing_humid_justifiability", "feature_croaking_japhetic_fifer", "feature_prothallium_burst_pledget", "feature_unassimilable_chunky_flattie", "feature_inhibiting_supernatural_runyon", "feature_resentful_eisteddfodic_assyrian", "feature_studded_premonitory_periostracum", "feature_dendritic_phytographic_skydiving", "feature_transpositional_undisciplinable_ancestor", "feature_redeemed_petiolar_lethe", "feature_enveloping_diabolic_serviette", "feature_excaudate_laboured_loquitur", "feature_unsympathetic_classic_abac", "feature_concise_tangy_bentham", "feature_freezing_unrenewed_gillray", "feature_intranational_sepulchral_clacton", "feature_mesarch_disentangled_catalytic", "feature_cauline_herpetic_videocassette", "feature_gravest_insincere_bronwen", "feature_plutocratic_cressy_plasterboard", "feature_massive_demisable_spouse", "feature_ugly_admissible_firm", "feature_reproachable_foliaged_lysozyme", "feature_sunburnt_sympodial_cookhouse", "feature_expended_evitable_darwinian", "feature_unreceipted_latest_lesser", "feature_necked_moresque_lowell", "feature_balaamitical_electropositive_exhaustibility", "feature_unvalued_untangled_keener", "feature_undisturbing_quadrifid_reinhardt", "feature_bucked_costume_malagasy", "feature_joint_unreturning_basalt", "feature_coordinate_shyer_evildoing", "feature_carunculate_discursive_hectare", "feature_cynic_unreckonable_feoffment", "feature_cnidarian_micrologic_sousaphone", "feature_unperceivable_unrumpled_appendant", "feature_dissolvable_chrismal_obtund", "feature_choosier_uncongenial_coachwood", "feature_grimmest_prostate_doctrinaire", "feature_granulative_uncritical_agostini", "feature_convalescence_deuteranopic_lemuroid", "feature_disintegrable_snakier_zion", "feature_thoughtful_accommodable_lack", "feature_basophil_urdy_matzo", "feature_repellant_unwanted_clarinetist", "feature_antimonarchist_ordainable_quarterage", "feature_hardback_saturnalian_cyclometer", "feature_mythic_florentine_psammite", "feature_serpentiform_incomplete_bessarabia", "feature_unappeasable_employed_photoelectron", "feature_unappreciated_humiliated_misapprehension", "feature_turbaned_excentric_rockery", "feature_aymaran_indusial_elodea", "feature_terpsichorean_hatable_glut", "feature_aeolic_downy_forefinger", "feature_undrowned_ascending_pungency", "feature_flagging_undenominational_gauffering", "feature_sworn_satanic_sprechstimme", "feature_atheist_pompeian_fogsignal", "feature_commotional_unhealable_berserk", "feature_grizzled_reformist_soberer", "feature_twiggier_fogged_prosodist", "feature_easterly_subtractive_faroese", "feature_looted_eleven_corpora", "feature_mensal_amusive_phosphorylase", "feature_shellier_dowable_chyme", "feature_daunting_nomenclatorial_facility", "feature_bedfast_primordial_hyponym", "feature_herpetologic_unjoyful_lodgepole", "feature_unlearned_plicate_megabit", "feature_cyclopedic_maestoso_daguerreotypist", "feature_disjunct_hardened_picturing", "feature_congolese_convenable_coolth", "feature_epigraphic_leucocratic_rutherford", "feature_hatched_myriad_biogen", "feature_gnomonic_fixative_vocalise", "feature_commo_flavored_epitomizer", "feature_exoergic_zoomorphic_burin"], "dexterity": ["feature_affettuoso_taxidermic_greg", "feature_lateral_confervoid_belgravia", "feature_geochemical_unsavoury_collection", "feature_guerrilla_arrested_flavine", "feature_undependable_stedfast_donegal", "feature_bijou_penetrant_syringa", "feature_lamarckian_tarnal_egestion", "feature_horticultural_footworn_superscription", "feature_small_cumulative_graywacke", "feature_incertain_catchable_zibet", "feature_woodier_slimmest_supplanter", "feature_conducive_underlying_snood", "feature_anencephalic_unattempted_pschent", "feature_bronchitic_miscible_inwall", "feature_sophistic_translucid_abutment", "feature_fortissimo_undrawn_ratio", "feature_appellant_verbalized_duckbill", "feature_sinister_footworn_tilling", "feature_subglobose_sleekier_calcaneum", "feature_addressable_intransitive_reconnoitrer", "feature_imitable_unnatural_samuel", "feature_wuthering_stinky_bimetallist", "feature_fungible_allotted_deterioration", "feature_saut_shalwar_culpability", "feature_dental_stormier_chape", "feature_irresponsible_unearthly_neat", "feature_alive_romansh_stinging", "feature_thermophile_noisette_swamper", "feature_corporatist_seborrheic_hopi", "feature_undisguised_unenviable_stamen", "feature_acclimatisable_unfeigned_maghreb", "feature_galactopoietic_luckiest_protecting", "feature_unheeded_stylar_planarian", "feature_preceptive_rushed_swedenborgian", "feature_sumerian_descendible_kalpa", "feature_jazziest_spellbinding_philabeg", "feature_dormie_sodden_steed", "feature_directoire_propositional_clydebank", "feature_cragged_sacred_malabo", "feature_idled_unwieldy_improvement", "feature_unmanaged_amative_grog", "feature_intime_impassible_ferrule", "feature_undevout_sonant_westerner", "feature_unlucky_hammered_pard", "feature_eucaryotic_centillionth_bastnaesite", "feature_mancunian_stalky_charmeuse", "feature_mettlesome_concussive_twinkling", "feature_unreduced_massive_hull", "feature_born_valvular_blob", "feature_setose_processed_crevice", "feature_associate_unproper_gridder"], "constitution": ["feature_floatiest_quintuplicate_carpentering", "feature_cuddlesome_undernamed_incidental", "feature_loony_zirconic_hoofer", "feature_indign_tardier_borough", "feature_fair_papal_vinaigrette", "feature_attack_unlit_milling", "feature_midmost_perspiratory_hubert", "feature_laminable_unspecified_gynoecium", "feature_ungenuine_sporophytic_evangelist", "feature_inconsiderate_unbooted_ricer", "feature_inured_conservable_forcer", "feature_glibber_deficient_jakarta", "feature_morbific_irredentist_interregnum", "feature_conjoint_transverse_superstructure", "feature_tingling_large_primordiality", "feature_phyllopod_unconstrainable_blubberer", "feature_deformable_unitary_schistosity", "feature_unprovisioned_aquatic_deuterogamy", "feature_equipped_undoubted_athanasian", "feature_soviet_zibeline_profiler", "feature_maxillary_orphic_despicability", "feature_clasping_fast_menstruation", "feature_babist_moribund_myna", "feature_stellular_paler_centralisation", "feature_cooled_perkiest_electrodeposition", "feature_differing_peptizing_womaniser", "feature_rankine_meaty_port", "feature_southernmost_unhuman_arbiter", "feature_unimpressed_uninflected_theophylline", "feature_subservient_wedged_limping", "feature_urticant_ultracentrifugal_wane", "feature_stoichiometric_unanswerable_leveller", "feature_cyanophyte_emasculated_turpin", "feature_unruly_salian_impetuosity", "feature_ataractic_swept_rubeola", "feature_burning_phrygian_axinomancy", "feature_dietetic_unscholarly_methamphetamine", "feature_vegetable_manlier_macaco", "feature_anthropoid_pithy_newscast", "feature_verifying_imagism_sublease", "feature_songful_intercostal_frightener", "feature_additive_untrustworthy_hierologist", "feature_translative_quantitative_eschewer", "feature_coseismic_surpassable_invariance", "feature_linear_scummiest_insobriety", "feature_ovine_bramblier_leaven", "feature_coalier_hircine_brokerage", "feature_undiverted_analyzed_accidie", "feature_favourable_swankiest_tympanist", "feature_refractory_topped_dependance", "feature_bustled_fieriest_doukhobor", "feature_isobilateral_olden_nephron", "feature_circassian_leathern_impugner", "feature_signed_ringent_sunna", "feature_cornute_potentiometric_tinhorn", "feature_veristic_parklike_halcyon", "feature_unwithered_personate_dilatation", "feature_wrought_muckier_temporality", "feature_rival_undepraved_countermarch", "feature_irrevocable_unlawful_oral", "feature_elohistic_totalitarian_underline", "feature_paraffinoid_flashiest_brotherhood", "feature_depauperate_armipotent_decentralisation", "feature_jamesian_scutiform_ionium", "feature_gambogian_feudalist_diocletian", "feature_moneyed_mesophytic_lester", "feature_purblind_autarkic_pyrenoid", "feature_yeasty_castilian_nicaragua", "feature_peak_interpretive_lahti", "feature_lithotomical_periodontal_systemization", "feature_martinique_tingliest_gynaecocracy", "feature_nymphomaniac_disappointing_greenaway", "feature_discombobulated_fourierism_histopathologist", "feature_granulitic_cordial_infield", "feature_piebald_unresisted_soldo", "feature_blushful_protean_granduncle", "feature_untired_flighty_tungstate", "feature_adjustable_ruffled_lumberjacket", "feature_toadyish_illiterate_famishment", "feature_inaccessible_radioactive_glassine", "feature_augean_contraceptive_subsequence", "feature_unbearded_rustiest_fiddlewood", "feature_contrasty_ablaze_canaster", "feature_uncharmed_rallying_ostracon", "feature_spicate_photolithographic_buckskin", "feature_disliked_undersized_vermiculation", "feature_costliest_heavenly_bovver", "feature_grasping_unmetrical_trollopian", "feature_interruptive_breached_protanomaly", "feature_reviving_mystifying_antwerp", "feature_sanctioned_sunny_lily", "feature_unmethodized_straining_heartburning", "feature_holometabolic_speeding_subinfeudatory", "feature_touring_urnfield_sequestrum", "feature_sunproof_unmurmuring_heliotype", "feature_aaronic_unexampled_arguer", "feature_tutelary_mangier_cryoscopy", "feature_depressant_blinded_yellowstone", "feature_saturated_protozoal_unsociability", "feature_nutant_legatine_fairfax", "feature_suety_mystagogical_islamization", "feature_defeasible_bustiest_trimetrogon", "feature_antitussive_vixenly_sacque", "feature_seminarial_catachrestic_i've", "feature_tangled_dropsical_uprooter", "feature_colourable_lapsable_foliage", "feature_heterotopic_choreographic_argentina", "feature_epithelial_apodictic_constructivism", "feature_resorbent_unmurmuring_humoring", "feature_uneducated_afraid_sip", "feature_spellable_quintic_idiotism", "feature_incredible_glimmering_hoydenism", "feature_scrappier_seen_phalanx", "feature_prostate_kernelly_chromogen", "feature_muggiest_explicit_barnardo", "feature_banal_microanalytical_posset", "feature_constrained_disused_perishable", "feature_apt_trimeter_bucketful", "feature_retardative_telencephalic_heme", "feature_quintuplicate_hortative_merionethshire", "feature_tetraploid_vorticose_mil", "feature_likeliest_exhaled_middlemarch", "feature_rhinological_downier_gamb", "feature_expectative_zonate_stockbroker", "feature_entrepreneurial_glumpiest_longhorn", "feature_recommendatory_prissy_flutter", "feature_blathering_unwell_despiser", "feature_intromittent_surer_pederasty", "feature_undealt_tonal_fictionalization", "feature_undrawn_oldish_deprivation", "feature_twistable_comely_licensee", "feature_overweening_incommunicable_pealing", "feature_phagocytic_humanistic_chappal", "feature_cyanotic_unblissful_aeration", "feature_indefeasible_abject_faucet", "feature_adolescent_anguilliform_staging", "feature_escheatable_miscreative_provence", "feature_coloratura_preclusive_micromillimetre", "feature_agape_untamable_towmond", "feature_underemployed_opiate_aube", "feature_winglike_hydropathic_sedimentology", "feature_conglomerate_amphipod_sewellel", "feature_virtuosic_afflicted_sumatra", "feature_chorionic_coated_undergraduette", "feature_tinkly_driftiest_maurya", "feature_pulsing_ionian_flatterer", "feature_downiest_spenserian_evadne", "feature_logistical_ahistorical_congregation", "feature_sensationalistic_puritan_mirador", "feature_steely_delusory_anesthesiologist", "feature_dissymmetric_stressed_stonewalling", "feature_lathery_uncarpeted_prolactin", "feature_coronate_congeneric_stockhausen", "feature_sane_disqualifying_whimperer", "feature_rifled_mum_ova", "feature_vibratory_prejudicial_quadrillion", "feature_enervated_tearier_septation", "feature_elasmobranch_braving_typhoid", "feature_autumn_prefectural_viscera", "feature_quaint_lyophobic_draper", "feature_nephritic_unrelievable_piperonal", "feature_mesarch_discreditable_calming", "feature_dendritic_phytographic_skydiving", "feature_differential_intercrossed_royalism", "feature_revelational_natty_nephralgia", "feature_subversive_guardable_kago", "feature_vulcanian_brimstony_autobiographer", "feature_unsoaped_waspier_much", "feature_vermillion_platyrrhine_demulsification", "feature_interchangeable_yellow_thinker", "feature_undermanned_transfusible_executive", "feature_jiggish_mechanized_patricide", "feature_tetrabasic_enervated_hemiparasite", "feature_chiseled_dastard_analyst", "feature_speediest_nautical_edge", "feature_fertilised_lakier_offside", "feature_parsonish_rutty_citronella", "feature_cyclone_disappointing_trollopean", "feature_sweatier_orbital_arrhenotoky", "feature_hydrokinetic_idyllic_archetype", "feature_mantuan_cricoid_responder", "feature_undesirable_candied_computist", "feature_tethered_exceptive_altimeter", "feature_mediative_sufferable_serosity", "feature_pinnatiped_unelected_irreverence", "feature_coated_transitory_oersted", "feature_epicontinental_centum_raine", "feature_desensitizing_distributive_bidder", "feature_syncytial_exterior_remora", "feature_sullied_vulval_disappointing", "feature_totipalmate_rightable_occultist", "feature_acanthoid_slimiest_decor", "feature_submontane_schmaltzy_piggyback", "feature_wannish_record_lunette", "feature_smuggled_scarabaeoid_fastball", "feature_sematic_helminthoid_tricentenary", "feature_untamed_contemplative_deism", "feature_municipal_curvier_hegelianism", "feature_pitchiest_dresden_barnard", "feature_korean_bassy_strewing", "feature_homelike_telltale_silvan", "feature_lacerable_backmost_vaseline", "feature_unimaginable_sec_kaka", "feature_goidelic_gobelin_ledge", "feature_incondite_undisappointing_telephotograph", "feature_concoctive_symmetric_abulia", "feature_anglophobic_unformed_maneuverer", "feature_required_bibliological_tonga", "feature_amoroso_wimpish_maturing", "feature_uncompelled_curvy_amerindian", "feature_tottery_unmetalled_codder", "feature_tachygraphical_sedimentological_mesoderm", "feature_adsorbed_blizzardy_burlesque", "feature_wistful_tussive_cycloserine", "feature_superjacent_grubby_axillary", "feature_biological_caprine_cannoneer", "feature_unreversed_fain_jute", "feature_unexalted_rebel_kofta", "feature_doggish_mouthwatering_abelard", "feature_forfeit_contributing_joinder", "feature_crimpier_gude_housedog", "feature_riskier_ended_typo", "feature_smaller_colored_immurement", "feature_conchal_angriest_oophyte", "feature_wariest_vulnerable_unmorality", "feature_cirsoid_buddhism_vespa", "feature_rid_conveyable_cinchonization", "feature_newfangled_huddled_gest", "feature_clandestine_inkiest_silkworm", "feature_cynic_unreckonable_feoffment", "feature_genoese_uncreditable_subregion", "feature_dexter_unstifled_snoring", "feature_orchitic_reported_coloration", "feature_stelliform_curling_trawler", "feature_athenian_pragmatism_isomorphism", "feature_abating_unadaptable_weakfish", "feature_partible_amphibrachic_classicism", "feature_cosy_microtonal_cedar", "feature_heedful_argyle_russianization", "feature_unhonoured_detested_xenocryst", "feature_sicker_spelaean_endplay", "feature_stratocratic_aerodynamic_herero", "feature_uneasy_unaccommodating_immortality", "feature_professional_platonic_marten", "feature_detrital_respected_parlance", "feature_saclike_hyphal_postulator", "feature_recent_shorty_preferment", "feature_scarcest_vaporized_max", "feature_spicier_unstripped_initial", "feature_hooly_chekhovian_phytogeographer", "feature_smouldering_underground_wingspan", "feature_phantasmal_extenuative_britain", "feature_sciurine_stibial_lintwhite", "feature_eucharistic_widowed_misfeasance", "feature_libratory_seizable_orlando", "feature_brackish_obstructed_almighty", "feature_translucid_neuroanatomical_sego", "feature_triangled_rubber_skein", "feature_vendean_thwartwise_resistant", "feature_preoral_tonsorial_souk", "feature_virescent_telugu_neighbour", "feature_undepreciated_partitive_ipomoea", "feature_southerly_assonant_amicability", "feature_cortical_halt_catcher", "feature_hermitical_stark_serfhood", "feature_deformable_productile_piglet", "feature_lentissimo_ducky_quadroon", "feature_happening_tristful_yodeling", "feature_geomedical_imbued_clunk", "feature_unadjusted_dissectible_warley", "feature_demountable_unprejudiced_neighbourhood", "feature_twisted_saronic_necrologist", "feature_celebratory_assayable_carlisle", "feature_cheerful_aphidian_orchestrion", "feature_transisthmian_inculcative_heldentenor", "feature_rampant_barren_sapodilla", "feature_often_undermanned_nudist", "feature_stannic_peevish_idocrase", "feature_biobibliographical_carnal_atomisation", "feature_depletory_cannular_automatism", "feature_collectable_distinguishing_dichroite", "feature_garlicky_allopatric_sarcocarp", "feature_whiskered_unobjectionable_quintet", "feature_enteric_booked_flexography", "feature_inlaid_defensible_gladiator", "feature_natal_scalloped_edwardianism", "feature_sphygmic_young_latium", "feature_infested_feathered_pen", "feature_monosyllabic_homey_omicron", "feature_thrasonical_subaltern_inoculation", "feature_exuberant_helicoidal_baldachin", "feature_vapourish_ichthyotic_causerie", "feature_intromittent_evasive_swordcraft", "feature_augmentable_scriabin_fortnight", "feature_exoergic_unschooled_lipid", "feature_inhospitable_baked_elopement", "feature_grizzled_reformist_soberer", "feature_intracardiac_circumfluent_pepper", "feature_contaminable_exilic_girandole", "feature_lowered_toric_charmeuse", "feature_altruistic_congenital_disinflation", "feature_rowdyish_overcritical_digression", "feature_peccant_zanier_undersigned", "feature_entozoic_adolescent_asci", "feature_antistatic_cabbagy_bluecoat", "feature_uncropped_tipsier_postulator", "feature_diocesan_reinvigorated_ebullience", "feature_axonometric_unkindly_sienna", "feature_ultra_unpolluted_adsorbent", "feature_bespangled_prim_might", "feature_comely_typal_softie", "feature_adherent_judaic_gerry", "feature_unelected_authorized_lucia", "feature_unreproved_cultish_glioma", "feature_ago_hypocritical_codeclination", "feature_proteinic_marcan_anxiety", "feature_endermatic_toasted_donald", "feature_restricting_unghostly_tapir", "feature_coequal_ambient_philopena", "feature_heliotypic_deprivative_behavior", "feature_unpriced_sniffiest_marvel", "feature_diphthongal_unvisored_knothole", "feature_jugoslav_cultured_tinct", "feature_gravettian_groveling_crooning", "feature_admissive_jaggiest_yabby", "feature_millenary_aliquot_hangdog", "feature_benzal_sprucest_taler", "feature_humpbacked_tribrachic_cosmotron", "feature_hated_twiggiest_mash", "feature_impeded_propagandist_darer", "feature_phylogenetic_paramount_caperer", "feature_easterly_predicable_enclosure", "feature_dyspeptic_unobstructive_rewriting", "feature_iridic_vellum_invective", "feature_bearable_sacrificial_sewer"], "wisdom": ["feature_froggier_unlearned_underworkman", "feature_peninsular_pulsatile_vapor", "feature_bally_bathymetrical_isadora", "feature_skim_unmeant_bandsman", "feature_kinky_benzal_holotype", "feature_ruptured_designing_interpolator", "feature_hierologic_expectable_maiolica", "feature_boiling_won_rama", "feature_lovelorn_aided_limiter", "feature_bratty_disrespectable_bookstand", "feature_mightier_chivalric_kana", "feature_overstrung_dysmenorrheal_ingolstadt", "feature_rose_buttoned_dandy", "feature_recipient_perched_dendrochronologist", "feature_spikier_ordinate_taira", "feature_mercian_luddite_aganippe", "feature_faint_consociate_rhytidectomy", "feature_unpressed_mahratta_dah", "feature_gleaming_monosyllabic_scrod", "feature_unyielding_dismal_divertissement", "feature_singhalese_cerographical_ego", "feature_agaze_lancinate_zohar", "feature_wally_unrotted_eccrinology", "feature_unforgivable_airtight_reinsurance", "feature_uninforming_predictable_pepino", "feature_pluviometrical_biannual_saiga", "feature_flawy_caller_superior", "feature_narcotized_collectivist_evzone", "feature_tallish_grimier_tumbrel", "feature_partitive_labyrinthine_sard", "feature_inhospitable_necked_duckbill", "feature_stolid_unhacked_schoolgirl", "feature_frogged_slightest_patmore", "feature_fascial_biserrate_pout", "feature_coercible_fecal_steradian", "feature_inadequate_unisex_internationalisation", "feature_darkened_campanulate_decerebrate", "feature_nephritic_grammatical_lithograph", "feature_lenient_electrothermal_phoenix", "feature_kingly_gemmological_electrodynamometer", "feature_applausive_forgettable_mishanter", "feature_unconfessed_paltry_finn", "feature_botchier_universalistic_nullifier", "feature_excursive_slaggy_confutation", "feature_tearing_inharmonic_employee", "feature_exhilarative_agleam_hebron", "feature_maigre_twinkling_overstand", "feature_handled_crescent_ciselure", "feature_apprehensible_assuring_schappe", "feature_trifling_sleety_amylase", "feature_jammed_stearic_gaper", "feature_biosynthetic_wambly_cullender", "feature_defective_sectional_stenotype", "feature_unworked_tribadic_catalyst", "feature_manufactured_nodal_seeking", "feature_asphyxiated_peaceful_effleurage", "feature_trad_unreduced_banian", "feature_fogyish_cruciate_starter", "feature_unpainted_censual_pinacoid", "feature_isoseismic_rhinocerotic_narceine", "feature_aleatory_phallic_swingtree", "feature_tragical_rainbowy_seafarer", "feature_chaliced_evolutional_street", "feature_legged_spatiotemporal_basalt", "feature_obligate_quadruplication_feathering", "feature_interpenetrative_boustrophedon_proudhon", "feature_pulmonic_bladed_affray", "feature_undisguised_photoelectric_floorboard", "feature_sodding_choosy_eruption", "feature_perverted_unapproving_sawyer", "feature_etched_furry_biriani", "feature_financed_striped_libertarian", "feature_pudendal_unterrifying_hagdon", "feature_standardized_rosiny_suslik", "feature_exploding_delectable_aril", "feature_hemihedral_fumed_marquisette", "feature_disillusive_saltant_placidity", "feature_squirarchal_bioplasmic_delay", "feature_gathered_owlish_judgment", "feature_dichotomic_tenpenny_myotonia", "feature_unapprehensive_thickety_etherification", "feature_unweary_avionic_claudine", "feature_satisfied_aymaran_enterotomy", "feature_indentured_insuperable_spider", "feature_gravimetric_ski_enigma", "feature_balmiest_spinal_roundelay", "feature_exertive_unmodernised_scaup", "feature_rude_booziest_ilium", "feature_footling_unpuckered_lophophore", "feature_thorniest_laughable_hindustani", "feature_hotter_cattish_aridity", "feature_developing_behind_joan", "feature_ectodermal_mandaean_saffian", "feature_inserted_inconvertible_functioning", "feature_drizzling_refrigerative_imperfection", "feature_smutty_prohibited_sullivan", "feature_productile_auriform_fil", "feature_accommodable_crinite_cleft", "feature_clipped_kurdish_grainer", "feature_dustproof_unafraid_stampede", "feature_neutered_postpositive_writ", "feature_twelve_haphazard_pantography", "feature_donsie_folkish_renitency", "feature_agee_sold_microhabitat", "feature_unutterable_softening_roper", "feature_seaboard_adducent_polynesian", "feature_liftable_direful_polyploid", "feature_objective_micro_langton", "feature_strip_honoured_trail", "feature_unsheltered_doughtiest_episiotomy", "feature_prefigurative_downstream_transvaluation", "feature_holy_chic_cali", "feature_huggable_interim_doline", "feature_tinkliest_unstuffy_manhunt", "feature_parturient_liberian_gamal", "feature_circulating_abolition_ethyne", "feature_ideological_trinal_rebuttal", "feature_figurative_uncertificated_indigent", "feature_improbable_pouched_gaitskell", "feature_unhazarded_droning_bellow", "feature_monarchic_blah_cellarman", "feature_walnut_sceptical_crystallization", "feature_quodlibetic_enrapt_miscalculation", "feature_doctrinal_viewier_dentary", "feature_careworn_motivational_requisite", "feature_psycholinguistic_junoesque_central", "feature_revolting_pharmacological_notability", "feature_unheeding_tauromachian_ballup", "feature_octopod_skirting_jurat", "feature_precursory_catching_inertia", "feature_bellicose_lunatic_glorification", "feature_undebauched_cobaltic_guerrilla", "feature_dysgenic_putrefied_nosegay", "feature_occurrent_suggestible_doubter", "feature_dada_draughtiest_cinchonisation", "feature_syndicalistic_epaxial_caldarium", "feature_rubbliest_cinnamic_gioconda", "feature_enervated_porose_microfarad", "feature_unalloyed_carminative_supercargo", "feature_mini_caressive_mantuan"], "agility": ["feature_unrelenting_intravascular_mesenchyme", "feature_scissile_dejected_kainite", "feature_ruthenic_peremptory_truth", "feature_digressive_ratty_supernatant", "feature_multipolar_syncopated_ambrotype", "feature_flamier_confusing_dithering", "feature_reverable_sunk_quiet", "feature_undrilled_wheezier_countermand", "feature_fearsome_merry_bluewing", "feature_entopic_interpreted_subsidiary", "feature_revitalizing_rutilant_swastika", "feature_carbuncled_athanasian_ampul", "feature_unransomed_unhealthier_excuser", "feature_milkier_gassy_pincushion", "feature_comprisable_commensurable_cyrenaic", "feature_antic_telekinetic_centrifuge", "feature_bearish_lesser_bloodstain", "feature_aquaphobic_paradisal_isagoge", "feature_sound_overabundant_agnomen", "feature_unlidded_chattier_usufructuary", "feature_agricultural_uranic_ankerite", "feature_bimanual_godly_witloof", "feature_anxiolytic_placatory_sextile", "feature_unsensing_enterprising_janissary", "feature_diffusive_unaccompanied_clubability", "feature_cistic_predeterminate_blackburn", "feature_pakistan_swirling_dystonia", "feature_bioplasmic_amended_iodism", "feature_defiled_feudalist_stonewaller", "feature_obtuse_waggly_entrancement", "feature_dysuric_permeated_makeweight", "feature_wordiest_babist_stackyard", "feature_foldaway_supernumerary_clubhouse", "feature_endoplasmic_inwrought_percival", "feature_checkered_accoutred_marjoram", "feature_lowery_transcribed_muffin", "feature_profaned_exothermal_orczy", "feature_bursarial_southmost_kaduna", "feature_elaborate_burning_drunkard", "feature_pardonable_ungraceful_bedazzlement", "feature_unholy_residential_anabaptism", "feature_uremic_trussed_grater", "feature_shrinelike_introverted_eagre", "feature_predominant_unmown_concealing", "feature_violated_telic_tuning", "feature_brief_optimistic_consentaneity", "feature_humanlike_urinant_snuffle", "feature_dividual_kufic_militarism", "feature_subminiature_catchable_classic", "feature_compatriotic_billion_revere", "feature_desired_thallophytic_brickfielder", "feature_hydriodic_metallurgic_stauroscope", "feature_stockish_overland_potentiation", "feature_stalagmitic_jacobethan_campanologist", "feature_theocentric_shameful_quintuplet", "feature_wombed_liberatory_malva", "feature_diametral_inflatable_editorialization", "feature_frowsier_productional_exemplification", "feature_acetose_periotic_coronation", "feature_irregular_sotted_biomedicine", "feature_directive_bioplasmic_skua", "feature_disparate_acellular_pictish", "feature_trimestrial_unsuspecting_guadeloupe", "feature_epinastic_sycophantical_satinwood", "feature_gaga_clinched_islamization", "feature_afoul_valvate_faery", "feature_pulsing_patrimonial_wame", "feature_underdeveloped_incomprehensible_traveller", "feature_polyphonic_superordinary_proximation", "feature_subjacent_repressive_biliverdin", "feature_nasofrontal_hornier_sterigma", "feature_apprentice_acheulian_extractability", "feature_gandhian_discretional_cricoid", "feature_nonagenarian_roundish_publication", "feature_togate_unbailable_door", "feature_setose_quodlibetical_stichic", "feature_true_legendary_shote", "feature_normal_urochordal_proffer", "feature_daring_telial_airspeed", "feature_spare_lingulate_withering", "feature_inclined_starchy_praseodymium", "feature_francophone_lattermost_spohr", "feature_uninvested_unwishful_scoria", "feature_pulverable_unpolitical_bathometer", "feature_isochronal_incorrect_desman", "feature_unpoisoned_migratory_uri", "feature_chatty_circumambient_patripassian", "feature_goyish_riparian_recipient", "feature_intramuscular_nummulitic_wildcatter", "feature_diatonic_duplex_bunny", "feature_fewest_held_giving", "feature_hyperthermal_deflationary_fasting", "feature_penned_insufficient_cartel", "feature_permed_steady_adminicle", "feature_prepotent_divorced_taffy", "feature_urodele_miffier_chagall", "feature_lower_legalism_stane", "feature_unpreached_pickiest_lint", "feature_ablest_mauritanian_elding", "feature_sliced_cuneal_anouilh", "feature_bifocal_disposable_clacton", "feature_splashier_conservant_ultramarine", "feature_fourieristic_allied_mugwumpery", "feature_headiest_unguessed_religion", "feature_nonnegotiable_errant_soya", "feature_substantiated_denatured_hadn't", "feature_optical_kempt_aisle", "feature_terroristic_tripersonal_pashm", "feature_herniated_exasperate_victorian", "feature_domanial_shellproof_rationing", "feature_prelingual_impracticable_plagiocephaly", "feature_ironclad_coppery_labour", "feature_tineal_premarital_rya", "feature_antimonial_unsold_hairdo", "feature_guttate_russian_greenhead", "feature_trespassing_unmacadamized_villeneuve", "feature_pulmonate_descendant_epiblast", "feature_imperialist_slovenly_licensor", "feature_illuvial_algebraic_modem", "feature_acquisitive_lengthening_matron", "feature_wetter_unbaffled_loma", "feature_unconjugal_chiropodial_amorosity", "feature_third_discreet_solute", "feature_unbarking_apolitical_hibernian", "feature_encysted_conventionalized_dematerialization", "feature_dominant_unreducible_iota", "feature_improvable_waniest_lesson", "feature_supererogatory_unleisured_kitling", "feature_sellable_supervenient_immobilism", "feature_chasmed_tergal_spencerian", "feature_spectacled_idiosyncratic_macula", "feature_unconstitutional_quadruped_carbine", "feature_displayed_denatured_fosterer", "feature_scalding_assumptive_sentimentalist", "feature_sounded_inescapable_chalybeate", "feature_circumspective_daughterly_brubeck", "feature_mimetic_sprawly_flue", "feature_inductile_umbrian_wallah", "feature_ineloquent_bihari_brougham", "feature_shakespearean_alpha_constituent", "feature_marxian_plated_refrigeration", "feature_amative_irresponsive_flattie", "feature_intermissive_coronal_reinsertion", "feature_dwarfish_isochronal_amateur", "feature_polyphyletic_unplumed_pandiculation"], "serenity": ["feature_honoured_observational_balaamite", "feature_polaroid_vadose_quinze", "feature_untidy_withdrawn_bargeman", "feature_genuine_kyphotic_trehala", "feature_unenthralled_sportful_schoolhouse", "feature_divulsive_explanatory_ideologue", "feature_ichthyotic_roofed_yeshiva", "feature_waggly_outlandish_carbonisation", "feature_floriated_amish_sprite", "feature_iconoclastic_parietal_agonist", "feature_demolished_unfrightened_superpower", "feature_styloid_subdermal_cytotoxin", "feature_ironfisted_nonvintage_chlorpromazine", "feature_drier_worshipping_hetairist", "feature_incredible_plane_sacque", "feature_inducible_home_immovability", "feature_feral_telling_marquessate", "feature_agitato_unlineal_perspicacity", "feature_turanian_satiable_millicent", "feature_girlish_uncoated_shammy", "feature_runniest_unstaying_toom", "feature_smashed_gynaecoid_septa", "feature_humiliating_numerate_goldminer", "feature_steadier_untrenched_bernstein", "feature_battled_premillennial_omelette", "feature_hefty_hesitant_mantissa", "feature_waxing_jaggy_bondswoman", "feature_hidden_blue_bibber", "feature_marginal_irredeemable_neat", "feature_baptist_undelayed_mannerism", "feature_eruciform_scorbutic_overkill", "feature_lentissimo_zymolytic_earwig", "feature_unreprimanded_evocable_briard", "feature_unqualifying_pursuant_antihistamine", "feature_crisscrossed_audible_hafiz", "feature_ugrian_schizocarpic_skulk", "feature_associable_additional_bough", "feature_doggone_seeable_mask", "feature_interconnected_correlatable_exogamy", "feature_blind_concordant_tribalist", "feature_strigose_rugose_interjector", "feature_binding_lanky_rushing", "feature_carolean_tearable_smoothie", "feature_nappiest_unportioned_readjustment", "feature_sarmatia_foldable_eutectic", "feature_plum_anemometrical_guessing", "feature_gubernacular_liguloid_frankie", "feature_castigatory_hundredfold_hearthrug", "feature_pennsylvanian_sibylic_chanoyu", "feature_unreaving_intensive_docudrama", "feature_relinquished_incognizable_batholith", "feature_indusiate_canned_cosh", "feature_teased_pinpoint_grant", "feature_periclean_proportionable_amaranth", "feature_lithuanian_fabianism_pedagogy", "feature_unamenable_prevalent_trilobite", "feature_intermingled_reedier_rookery", "feature_jangly_weedier_bhang", "feature_bubbling_pedestrian_convection", "feature_supportive_explanatory_powder", "feature_ruttier_freakier_perversion", "feature_duskier_wispiest_midwesterner", "feature_martial_hallowed_incorruptibility", "feature_exhaling_awkward_inhabitant", "feature_cristate_desirable_chime", "feature_aguish_commissioned_tessitura", "feature_desecrated_antiseptic_pirog", "feature_juvenile_carlish_betel", "feature_synodal_thornier_zila", "feature_delayed_reluctant_castro", "feature_olden_enchained_leek", "feature_ochlocratical_hemiparasitic_brothel", "feature_syndicalistic_osteal_matriarchy", "feature_objectivist_adaptive_charr", "feature_accoutred_fluviatile_vivification", "feature_exploitative_jetty_oujda", "feature_suppler_faraway_hatchback", "feature_petaline_circumscribable_sartre", "feature_unadaptable_floored_styptic", "feature_hurtling_wizened_cockade", "feature_diarrhoeic_relieved_scutter", "feature_profaned_obsequent_urology", "feature_unrecognisable_ultrabasic_corporeity", "feature_reposeful_apatetic_trudeau", "feature_seismograph_molybdic_requisition", "feature_suppressed_unremovable_telephone", "feature_immovable_apiarian_joke", "feature_exhibitionist_bicuspidate_goalpost", "feature_genic_knobbed_malacologist", "feature_warmish_unspiritualizing_desideratum", "feature_driven_preserving_spectroheliogram", "feature_eighteen_kafka_segno", "feature_insistent_presageful_deist", "feature_cavitied_alleviatory_neuk", "feature_crispy_neighboring_jeffersonian"], "sunshine": ["feature_differing_lenticular_gossiping", "feature_ottoman_metaphysical_parathion", "feature_variolate_reducible_sweet", "feature_jumpiest_clattering_pentstemon", "feature_sylphid_maladaptive_franc", "feature_aguish_commissioned_tessitura", "feature_desecrated_antiseptic_pirog", "feature_juvenile_carlish_betel", "feature_synodal_thornier_zila", "feature_delayed_reluctant_castro", "feature_olden_enchained_leek", "feature_ochlocratical_hemiparasitic_brothel", "feature_syndicalistic_osteal_matriarchy", "feature_objectivist_adaptive_charr", "feature_accoutred_fluviatile_vivification", "feature_exploitative_jetty_oujda", "feature_suppler_faraway_hatchback", "feature_petaline_circumscribable_sartre", "feature_unadaptable_floored_styptic", "feature_hurtling_wizened_cockade", "feature_diarrhoeic_relieved_scutter", "feature_profaned_obsequent_urology", "feature_unrecognisable_ultrabasic_corporeity", "feature_reposeful_apatetic_trudeau", "feature_seismograph_molybdic_requisition", "feature_glandered_steamtight_transform", "feature_peachier_unswallowed_gil", "feature_yoruban_purplish_directoire", "feature_unrhymed_synoptistic_combine", "feature_voyeuristic_ineffable_yelling", "feature_unluckier_zonary_invalidism", "feature_unshockable_diffused_granadilla", "feature_diactinic_distant_populariser", "feature_campodeid_myasthenic_merrymaking", "feature_jilted_epagogic_olivier", "feature_typic_anucleate_caecum", "feature_fumed_pivotal_oscine", "feature_preterite_antediluvian_parasailing", "feature_folkish_mononuclear_granitization", "feature_sixfold_tipsier_roup", "feature_conductive_tribunicial_supertitle", "feature_rhinencephalic_goofier_fulah", "feature_ghastlier_treed_davy", "feature_seventy_bearish_kleptomaniac", "feature_hydrometrical_quadricentennial_medial", "feature_blind_alabaman_brabble", "feature_stockiest_untransmitted_greening", "feature_polypoid_taken_upgrader", "feature_hygroscopic_clithral_leakage", "feature_bolivian_astringent_didapper", "feature_intersectional_inseminated_undersigned", "feature_idyllic_spectacled_togue", "feature_antiskid_circumlocutional_ogdon", "feature_overdone_raspiest_halcyon", "feature_coagulatory_bathymetrical_pergamum", "feature_shorthand_elemental_overall", "feature_retractile_sayable_physic", "feature_calefactive_baculiform_frogfish", "feature_unvaccinated_fretted_phaeacian", "feature_sheen_deteriorating_carnassial", "feature_federate_ungoverned_nitwit", "feature_lachrymatory_welcomed_flying", "feature_authorised_new_macrosporangium", "feature_hymnal_suberic_sulphurator", "feature_unflinching_bustled_pehlevi", "feature_foraminal_structured_corruption", "feature_humdrum_unbusinesslike_corrupter", "feature_unverifiable_girly_lashing", "feature_crackly_ripuarian_parure", "feature_tenebrific_antefixal_explicator", "feature_bejewelled_effaceable_urate", "feature_ventriloquistic_relegable_optometer", "feature_plantable_integumentary_roper", "feature_depilatory_tin_trinity", "feature_unespied_papuan_kaduna", "feature_undermasted_portly_divinity", "feature_patronal_hussite_stroking", "feature_trad_glairiest_advocaat", "feature_octaval_thieving_knosp", "feature_unwrapped_flashier_luggie", "feature_unweathered_inspirative_shoestring", "feature_indiscriminate_cuneatic_soundman", "feature_worrisome_galvanic_cockneydom", "feature_mammonistic_pulsed_welter", "feature_nonagon_quietening_bressummer", "feature_paragenetic_traditional_madeline", "feature_collapsable_sinful_cassareep", "feature_preceding_perturbing_radii", "feature_hypogeal_unburied_embraceor", "feature_hobbesian_infrasonic_subjugator", "feature_phobic_emptied_esteem", "feature_meteorological_ritzier_diffractometer", "feature_bulbed_pinioned_serfdom", "feature_unshockable_lamelliform_paederast", "feature_sly_dumpier_gynomonoecism", "feature_tramontane_malevolent_endoscopy", "feature_fronded_interferential_quadrat", "feature_exocrine_early_consistory", "feature_clad_refusable_trochanter", "feature_coppiced_sign_staysail", "feature_saleable_unreprimanded_sacrosanctity", "feature_tingliest_prefatory_trapshooter", "feature_imperishable_mightiest_emulsoid", "feature_rococo_unexceptional_tropicbird", "feature_couthie_inexplicit_reinfection", "feature_unbiased_judicatory_potentiometer", "feature_tenty_filigree_bengaline", "feature_chasmed_unexhausted_cryptogram", "feature_wealthy_asinine_beduin", "feature_conflictive_cosher_iranian", "feature_ane_unprompted_columbary", "feature_delicate_scant_shore", "feature_gawky_biracial_jephthah", "feature_bucolic_sedative_quivering", "feature_cathodic_unstigmatised_rockford", "feature_deliquescent_pelvic_functionary", "feature_maternal_knobbier_dialecticism", "feature_blowzier_sleepiest_verifier", "feature_uncurtailed_sabaean_ode", "feature_crosscut_rompish_osteopathist", "feature_piquant_israeli_sperm", "feature_statistical_beefier_fluoride", "feature_docked_sufistic_oropharynx", "feature_stubbled_anatolian_pout", "feature_whitish_uncontestable_palermo", "feature_expectative_intimidated_bluffer", "feature_creaking_unsophisticated_clavier", "feature_prissy_counsellable_keg", "feature_unreposing_mellifluent_dindle", "feature_cowled_knottiest_megilp", "feature_scurry_assuasive_internode", "feature_domesticable_embowed_ommatidium", "feature_glycogenetic_meagre_ratite", "feature_valved_streamier_gloucestershire", "feature_neoclassicist_buttressed_preface", "feature_yoruban_unapplied_tawse", "feature_electrophysiological_blearier_deconsecration", "feature_nepotic_overreaching_itinerary", "feature_scrawny_wavy_pheon", "feature_incurrent_homeopathic_overcapacity", "feature_personate_sublunar_eugene", "feature_surmountable_proved_compeer", "feature_pucka_bloomy_mycophagist", "feature_shattering_indented_dolby", "feature_unkept_warier_yucca", "feature_annihilative_yoruban_wile", "feature_kittenish_resigned_sequencer", "feature_unworked_charybdian_managing", "feature_centenary_dismounted_general", "feature_aswarm_vengeful_bilge", "feature_immemorial_lousy_wishbone", "feature_calcaneal_phenological_probing", "feature_swashbuckling_unnative_rouser", "feature_lousiest_neanderthal_hypha", "feature_segregable_blasting_inscription", "feature_viridescent_choking_pinto", "feature_julienne_interludial_noradrenaline", "feature_unexplained_orthorhombic_kenny", "feature_aspheric_cloudy_restorationism", "feature_epistatic_absent_slayer", "feature_geosynclinal_concluding_nookie", "feature_urnfield_linty_strawman", "feature_attractable_lawgiver_mbujimayi", "feature_sematic_underpeopled_dueller", "feature_closing_branchy_kirman", "feature_stuffed_unabashed_biretta", "feature_invocatory_unplumbed_assessorship", "feature_speaking_burdensome_you'd", "feature_azimuthal_disconcerted_bock", "feature_topping_over_anecdotage", "feature_papillose_unprevailing_conductivity", "feature_polluted_extraverted_limey", "feature_anagrammatical_lignitic_morel", "feature_prefatorial_empirical_undertenant", "feature_hussite_frecklier_bransle", "feature_won_stalwart_eisenstein", "feature_indulgent_amphibrachic_surrebutter", "feature_hypnagogic_wedded_religionist", "feature_dormant_samariform_elucidator", "feature_geostationary_manky_teutonisation", "feature_unabbreviated_unteamed_krasnoyarsk", "feature_deontological_sidearm_alder", "feature_saprogenic_quadrilateral_chook", "feature_hagiographical_fertile_disestablishment", "feature_moonlit_hundred_conferrer", "feature_vertebrate_anisotropic_chewer", "feature_chadic_allotropic_delirium", "feature_quartile_athletic_schwarzkopf", "feature_pileate_accusatival_immunogen", "feature_star_roomier_mapping", "feature_deistical_intractable_veadar", "feature_ontological_secondary_analcite", "feature_vaccinated_compellable_schizont", "feature_guarded_rotational_flotow", "feature_monocyclic_unrejoiced_haematoxylon", "feature_retributive_unconformable_hairpin", "feature_annoyed_obsessive_watch", "feature_sallow_jaculatory_galactopoietic", "feature_hyphenic_pudendal_defeasibility", "feature_fibrillar_crural_persimmon", "feature_caudated_consuetudinary_bratislava", "feature_incommensurate_stung_impassibility", "feature_rumbly_unlabelled_insurant", "feature_festive_pewter_peeper", "feature_corned_moderating_inaudibility", "feature_gasteropod_weird_virucide", "feature_venerating_reduplicate_licensee", "feature_unespied_vorticose_valour", "feature_opposing_intracardiac_delimitation", "feature_manlier_dopiest_chiefdom", "feature_clumsiest_doctoral_monk", "feature_unsectarian_unuseful_opiate", "feature_interconnected_greige_mohammedan", "feature_petrogenetic_tapetal_pavior", "feature_surpliced_unachievable_nubecula", "feature_unsubscribed_pyknic_thalweg", "feature_primogenital_paralytic_minx", "feature_alfresco_unresolvable_kashmir", "feature_shoreward_haustellate_acorn", "feature_photic_untunable_father", "feature_immensurable_myrmecological_angler", "feature_telencephalic_assumed_loran", "feature_other_peatier_raymond", "feature_unkingly_protochordate_confluent", "feature_falcate_concurrent_kwa", "feature_leptosporangiate_perceptive_urari", "feature_sophistical_canty_mastersinger", "feature_impetratory_interrogative_sangaree", "feature_selachian_gestic_dapple", "feature_much_bandoliered_refundment", "feature_organic_hellenic_venesection", "feature_swimmable_lumpish_baiting", "feature_gonococcic_ghostliest_excuser", "feature_hundredth_hymnal_negative", "feature_unnoted_frowzier_protest", "feature_delimited_bolted_canner", "feature_offhand_reinforced_bump", "feature_vendible_unprocurable_lignum", "feature_unabbreviated_craftier_conodont", "feature_delineable_microsomal_foeman", "feature_snuggest_counterclockwise_desolater", "feature_perspectivist_nondestructive_haemocoel", "feature_demonstrated_wordiest_globulin", "feature_wrinkliest_unmaintainable_usk", "feature_unhurt_centroidal_whimsey", "feature_select_comprehensible_spanish", "feature_spun_isoclinal_agate", "feature_violated_yonder_skipper", "feature_propagative_unloving_carioca", "feature_metal_bunchier_uranism", "feature_next_fusile_mentum", "feature_lamented_dead_incalculability", "feature_trusted_painful_hetty", "feature_driest_marmoreal_industrialisation", "feature_hirsute_corkier_beldame", "feature_accretive_sorrier_skedaddle", "feature_unmanned_connecting_sadducee", "feature_tiresome_scary_didn't", "feature_uncongenial_developmental_underdevelopment", "feature_unpopular_promissory_liturgiologist", "feature_concentric_gubernatorial_grandeur", "feature_roasted_fortified_transfiguration", "feature_unclean_pediculate_cymbal", "feature_antimonic_conglomerate_demolishing", "feature_juvenile_tergal_pseudomorph", "feature_fine_drouthiest_nekton", "feature_tumbling_false_anagoge", "feature_flabbergasted_evidenced_aire", "feature_instinct_reproved_capitate", "feature_imputable_aymaran_thruway", "feature_overfed_segmented_exedra", "feature_awol_choriambic_hankie", "feature_astatic_foliate_whitsun", "feature_clandestine_persistent_offertory", "feature_sivaistic_acinose_adult", "feature_ritual_torporific_ennui", "feature_unfaltering_peltate_diamorphine", "feature_agravic_incognoscible_gaddafi", "feature_arachnidan_hotter_fudge", "feature_elucidative_transversal_lawmaker", "feature_submersed_detectible_prospector", "feature_nutrimental_floatable_synthetizer", "feature_upstair_polycyclic_footsie", "feature_conceptional_flyaway_suburbanization", "feature_unhampered_attenuate_mot", "feature_supratemporal_pharmaceutical_ambassadorship", "feature_unordered_insertional_looter", "feature_geothermal_optional_octagon", "feature_overnight_fluent_trepan", "feature_necrotic_potamic_passionary", "feature_refutable_predatory_gesture", "feature_fold_straightforward_peacetime", "feature_biotic_harmonical_riser", "feature_highland_regent_dissector", "feature_shrunken_attractive_seigneur", "feature_whiggish_suffering_tonsillectomy", "feature_percurrent_balustraded_armlet", "feature_buccaneerish_cerulean_cetology", "feature_trotskyism_monozygotic_linlithgow", "feature_unreasoning_feudal_retarder", "feature_psychrometrical_inexpensive_opah", "feature_intranational_synovial_bish", "feature_attemptable_demotic_nestorianism", "feature_applicative_cometic_gimp", "feature_weighty_floriated_servomechanism", "feature_befitting_isogeothermic_diffidence", "feature_unmethodical_hipper_bergamo", "feature_kingly_understandable_matabeleland", "feature_feebler_asthmatic_rinse", "feature_stickier_theoretical_innovation", "feature_talkable_juicy_mitochondrion", "feature_emigrational_spryer_vaporosity", "feature_fluted_blightingly_sharpshooter", "feature_vorant_unsullied_sponsorship", "feature_predestinate_draconic_debug", "feature_aged_phylacterical_pusey", "feature_revisional_ablutionary_depression", "feature_yokelish_metapsychological_lunt", "feature_circumlunar_chaliced_seam", "feature_squallier_prototypal_dammar", "feature_cognate_elating_ravine", "feature_ethiopian_carminative_retentivity", "feature_alabamian_outlying_monitoring", "feature_byzantine_festinate_mannose", "feature_sleetier_sea_potamogeton"], "rain": ["feature_bumpier_maidenlike_chordata", "feature_moveable_hairiest_extinguishant", "feature_sallowy_confounding_trumping", "feature_demulcent_reachable_pteridologist", "feature_seventeenth_underfired_grimoire", "feature_extravehicular_hertzian_moo", "feature_frolicsome_candid_interambulacrum", "feature_antennal_intersidereal_sunn", "feature_credent_snuffly_apraxia", "feature_unpolarised_genal_premillenarian", "feature_veddoid_irritable_heidelberg", "feature_surefooted_quakier_constructivism", "feature_fidgety_shuddering_emperorship", "feature_unsandalled_uncertificated_rummy", "feature_anthracoid_unforeseen_arizonan", "feature_ulterior_flabbier_antimasque", "feature_thinkable_pledged_marten", "feature_groutier_inapposite_spindling", "feature_biodynamic_shintoist_tractarian", "feature_measured_fated_dogmatiser", "feature_battled_elephantine_paiute", "feature_spheric_assertive_auslese", "feature_submersed_unreturned_intermodulation", "feature_scorching_thai_subsystem", "feature_intramural_multivariate_bluebell", "feature_volatilized_labouring_raffinate", "feature_bouffant_unremoved_seascape", "feature_amplest_scyphiform_doing", "feature_bullied_hypostatic_bergen", "feature_copepod_ceratoid_jactation", "feature_preachy_unsatisfying_chaeta", "feature_unassailable_translational_rata", "feature_lurching_adjective_weekend", "feature_ovoid_bending_dispensator", "feature_hourlong_profane_hookey", "feature_diastatic_sorer_locative", "feature_auburn_reconstructional_nejd", "feature_resuscitable_romanticist_phellem", "feature_syndromic_rubblier_taxiway", "feature_cirrhotic_miffiest_cleansing", "feature_yorkist_authenticated_lotted", "feature_conjugative_surpliced_communicant", "feature_poikilothermic_ebony_wardenship", "feature_algonquin_performing_teetotalism", "feature_shuttered_knavish_salmonella", "feature_bawdiest_striking_gormand", "feature_eristic_documental_monochromatism", "feature_sensitive_incendiary_heraclid", "feature_conduplicate_levorotatory_sympathomimetic", "feature_digressional_multisulcate_paisley", "feature_scenographical_omnidirectional_selenium", "feature_dissipated_dressiest_bombast", "feature_bantam_matterful_hut", "feature_transhuman_diocesan_aston", "feature_declinatory_unfilmed_lutheranism", "feature_unwatered_dentate_unbelief", "feature_deflationary_hexaplaric_heterology", "feature_frightful_cercal_niccolite", "feature_radioactive_gigantean_oppilation", "feature_protochordate_aglimmer_dormouse", "feature_necessary_dreamy_bedside", "feature_indifferent_fungicidal_prescription", "feature_appassionato_censual_laverock", "feature_condyloid_hydriodic_synonymity", "feature_substitute_tamable_solum", "feature_robust_legatine_desk", "feature_intoed_pedigree_canful", "feature_divalent_centurial_hoya", "feature_guttering_supernatant_vernier", "feature_quadripartite_folded_mariachi", "feature_communicative_asphyxiated_catastrophist", "feature_zoophoric_underglaze_algin", "feature_cachinnatory_spumescent_acclimatization", "feature_minuscule_confusing_flaunter", "feature_metapsychological_inexcusable_manhunt", "feature_established_swift_tenia", "feature_skewbald_woodworking_haoma", "feature_edenic_hunched_megabuck", "feature_impennate_antistrophic_grocer", "feature_frore_mean_conscript", "feature_correspondent_orderly_personalisation", "feature_voluntary_dipteran_munich", "feature_thymic_formidable_misericord", "feature_blastoderm_jaspery_freeloader", "feature_unpalatable_mortifying_taurine", "feature_unanchored_scheming_demonstrator", "feature_brief_aforesaid_engulfment", "feature_insultable_perforate_raffle", "feature_rembrandtish_preclassical_deity", "feature_stoutish_zibeline_rentier", "feature_supersensual_unknown_alecto", "feature_unhesitating_governessy_kirchner", "feature_electroplate_unblenched_communion", "feature_zincky_unseemly_butt", "feature_hypnoid_unsurfaced_nonillion", "feature_cushitic_sequestered_tardigrade", "feature_misshapen_stochastic_tortilla", "feature_coraciiform_foreseeable_tutiorism", "feature_booming_venose_feudatory", "feature_phonematic_overdue_tabor", "feature_fanfold_tartarian_diamondback", "feature_floodlighted_apprentice_comstockery", "feature_instructional_confutative_shaktism", "feature_barish_slouchier_bullroarer", "feature_participating_unrecollected_braiding", "feature_conjugational_unamused_thrace", "feature_throbbing_pinchbeck_sememe", "feature_terrorful_unbaptised_tachogram", "feature_intercommunal_epitomical_geomagnetism", "feature_gala_beneficent_sedilia", "feature_colloid_frizzliest_poddy", "feature_unsurveyed_hymenal_cheapskate", "feature_amitotic_gonadial_submediant", "feature_embolismic_diastyle_raspberry", "feature_unvisored_bedraggled_bushel", "feature_iridescent_abiogenetic_sena", "feature_swanky_cupular_chaplainry", "feature_thumbed_diet_encephalograph", "feature_untarred_chiropodial_contagium", "feature_unimposing_theistic_hancock", "feature_bothered_dinky_eyesight", "feature_oblanceolate_macrobiotic_tightening", "feature_unconcealed_untaxed_oratory", "feature_uncluttered_hercynian_continuum", "feature_unreproving_capsian_decolourization", "feature_lemuroid_unwishful_mannequin", "feature_detectable_fogbound_dicastery", "feature_assuasive_wholesale_semele", "feature_unmellowed_unweakened_bibliopoly", "feature_warring_precise_doge", "feature_obbligato_crackbrained_wolverhampton", "feature_gushier_animistic_bohemian", "feature_uncorrupted_adducting_savin", "feature_left_retroflexed_underclassman", "feature_shelvy_egalitarian_cardialgia", "feature_sailing_viricidal_cowherd", "feature_creepiest_bicorn_gratification", "feature_damask_tabu_cobweb", "feature_radiate_quantifiable_chastity", "feature_detectable_balinese_mine", "feature_unsuiting_enuretic_milometer", "feature_geophytic_penitential_deutzia", "feature_offsetting_soled_desalinization", "feature_widowed_hellish_jaguarondi", "feature_libidinal_guardable_siderite", "feature_demonic_eocene_polygamy", "feature_unsubmerged_scathing_vapidity", "feature_ministrative_unvocalised_truffle", "feature_cirrose_rhaetic_londoner", "feature_protean_rubbery_bigener", "feature_denumerable_unsuccessive_unrealism", "feature_spatiotemporal_carthaginian_capture", "feature_veddoid_sport_psychobiology", "feature_oversimplified_expansionary_jitterbug", "feature_heterotrophic_anechoic_annexationist", "feature_mammoth_judiciary_honeypot", "feature_diversified_adventive_peridinium", "feature_perceptual_pausal_sheikdom", "feature_dizziest_insolvent_ctene", "feature_bathymetric_valiant_bahuvrihi", "feature_pronominal_billowing_semeiology", "feature_promulgated_tangled_nobleman", "feature_financed_sulphuretted_libertinage", "feature_cavicorn_transversal_peasant", "feature_pinnated_grasping_overcall", "feature_albinotic_ugly_remit", "feature_ungratified_filigree_dram", "feature_phrenitic_foldable_trussing", "feature_unashamed_sublimed_moulding", "feature_skim_expugnable_subception", "feature_undescended_crawly_armet", "feature_batholitic_intensional_interviewer", "feature_formidable_unrotted_craniotomy", "feature_enarched_assyrian_giuseppe", "feature_schlock_quaky_york", "feature_unshod_satiated_manioc", "feature_mammonistic_smeared_stigma", "feature_commendable_nicotined_banging", "feature_aloetic_aperiodic_dislocation", "feature_bloodshot_inexperienced_adductor", "feature_pronounceable_nonuple_cruller", "feature_ebracteate_autogenic_trimeter", "feature_crank_center_interweave", "feature_unpassable_wedgy_blossom", "feature_eclamptic_unblissful_dip", "feature_unfertilized_scaldic_partition", "feature_volumetrical_splenic_shoelace", "feature_matchmaking_polyatomic_foreboder", "feature_sacculate_inebriated_tamarack", "feature_thinking_grandfatherly_psychiatry", "feature_plangent_devoured_jarl", "feature_vital_pale_disassociation", "feature_eruciform_novice_thanker", "feature_liberticidal_subaqua_embassador", "feature_suspensory_unrecounted_transcendent", "feature_bamboo_nosier_phil", "feature_asinine_unsatiable_avion", "feature_societal_observational_pekingese", "feature_irresponsive_motherlike_enabler", "feature_pineal_translational_cleptomania", "feature_even_protecting_illuminance", "feature_laryngological_honour_artifice", "feature_smacking_unconsummated_wiggery", "feature_secular_hackneyed_latria", "feature_british_inspectional_presentment", "feature_indicial_caryatidal_kendal", "feature_miasmal_mozartian_gervase", "feature_decagonal_mozarabic_inclemency", "feature_undoubtful_soppiest_trigram", "feature_alert_eddic_semicylinder", "feature_peeved_abbatial_ante", "feature_katabolic_peridotic_ergotism", "feature_torporific_elastomeric_majesty", "feature_moneyed_usufruct_bismuth", "feature_prejudicial_catachrestical_discontinuance", "feature_size_interactive_liquefaction", "feature_paphian_octennially_limey", "feature_interlaced_tricyclic_microlite", "feature_symmetric_transmissive_calyptra", "feature_leukemic_hellenistic_economist", "feature_emphasized_confirming_clandestinity", "feature_russety_multidirectional_macaque", "feature_flagging_gadarene_barrymore", "feature_laggardly_prideful_turban", "feature_southern_investigative_carpology", "feature_shimmering_unsystematical_suzerainty", "feature_bristled_slender_transmutation", "feature_combinable_platiest_karin", "feature_ravaging_coalitional_boyer", "feature_amphitheatric_mineralized_overture", "feature_hellenic_rigid_moharram", "feature_staggering_spondaic_strindberg", "feature_unorthodox_tuneful_antilogy", "feature_grouchier_undoubting_sultana", "feature_unconjugal_deferrable_sheeting", "feature_fine_poky_friary", "feature_unpersuasive_phraseologic_turkmenistan", "feature_inappreciable_unmeriting_litre", "feature_token_east_victor", "feature_chilean_hobbesian_browsing", "feature_greasy_bloodier_subscription", "feature_unfadable_vaunting_soya", "feature_changed_proletarian_theodolite", "feature_cacophonic_sextan_liquescence", "feature_pinto_pesky_compaction", "feature_unimagined_radiographic_spinsterhood", "feature_seven_convertible_lixivium", "feature_unrecommended_acanthocephalan_gallicism", "feature_shotgun_attractive_bombshell", "feature_tumbling_gone_yawper", "feature_kirtled_cockiest_etaerio", "feature_fishable_ascendible_micky", "feature_electronegative_lactogenic_merc", "feature_surrogate_unmalleable_tasset", "feature_grave_prevenient_rheotrope", "feature_obovoid_hipped_vaporing", "feature_supersaturated_scalding_bribery", "feature_desired_eery_cypher", "feature_scalier_gracile_owenist", "feature_rawboned_bloodshot_cousinhood", "feature_lovable_record_phlegm", "feature_diphycercal_wrinklier_jewelfish", "feature_septilateral_parallactic_ngaio", "feature_phrenetic_visitorial_entrenchment", "feature_gloomful_uniaxial_tyrian", "feature_ponderable_faultiest_pfennig", "feature_next_moldered_paganism", "feature_unprohibited_chilliest_incurable", "feature_baric_troglodytic_deducibility", "feature_plentiful_remorseful_capacitor", "feature_virtuoso_gasping_studwork", "feature_rectilineal_stative_carousal", "feature_chastised_antitypical_palooka", "feature_atrial_retroactive_dolin", "feature_quadraphonic_pinnate_kouprey", "feature_gamest_zibeline_oakley", "feature_particulate_pericentral_refuse", "feature_existential_oecumenic_draco", "feature_rearing_midget_friedcake", "feature_surd_commutative_palliasse", "feature_aneurismal_set_hydranth", "feature_shock_exoskeletal_synagogue", "feature_bellied_umbilical_conglobation", "feature_isochimal_saving_combe", "feature_presbyopic_indiscreet_clancy", "feature_undismayed_rallentando_snooker", "feature_boyish_oily_sciurine", "feature_breeziest_religionism_synthetiser", "feature_imbecile_daimonic_endgame", "feature_expedited_yucky_anesthetic", "feature_sea_copyright_parsee", "feature_unnoticeable_clathrate_dairywoman", "feature_spectacular_unlisted_squalene", "feature_cloaked_taillike_usurpation", "feature_incorrigible_contaminate_monorhyme", "feature_alleviatory_sociopathic_photopia", "feature_alleged_weepier_tetanization", "feature_demoniacal_phylacterical_brach", "feature_mesmerized_springing_euchologion", "feature_daimonic_triennial_sweeping", "feature_overeager_pugilistic_diocletian", "feature_periosteal_fibrillose_eponym", "feature_subjective_crescentic_stereograph", "feature_inspective_unsolvable_subtangent", "feature_isodiametric_afraid_verderer", "feature_curvier_echinoid_leyden", "feature_loved_halcyon_rotifer", "feature_sear_bicorn_reorder", "feature_neonatal_undubbed_consigner", "feature_aneroid_cufic_prolonge", "feature_priced_choky_ishmaelite", "feature_returning_isoglossal_transmontane", "feature_mede_fogbound_triphenylmethane", "feature_joltier_mishnic_semiotician", "feature_lunisolar_depopulated_slaying", "feature_approximative_nuclear_readaptation", "feature_determinative_unsound_samizdat", "feature_kin_stellate_dogvane", "feature_agential_present_sclerotomy", "feature_plagal_distortive_pharyngeal", "feature_unperceptive_hypostatic_hibernicism", "feature_furrowed_enameled_mission", "feature_foaled_crutched_habitability", "feature_hempen_insubordinate_sarum", "feature_vixenish_nodose_phocomelia", "feature_discontent_sulfa_applicability", "feature_stylistic_pythagorean_pulley", "feature_poetic_chapped_refocusing", "feature_thermic_spectrographic_bend", "feature_vertebral_arboreal_beryllium", "feature_zincoid_peccant_greywacke", "feature_audible_scurrile_saltpeter", "feature_escharotic_humanistic_placebo", "feature_incorruptible_transpacific_ratine", "feature_careful_valid_picosecond", "feature_unimpeached_anguilliform_cymry", "feature_bibliomania_bodger_ensigncy", "feature_swelled_jugate_haystack", "feature_sonic_blond_redbreast", "feature_uncompanioned_interrelated_brimstone", "feature_unwilled_exhortative_gisarme", "feature_newfangled_irksome_sleigher", "feature_homely_unsmoothed_dubrovnik", "feature_sapphire_lyrate_christianism", "feature_interstadial_georgic_hellene", "feature_gentle_comminatory_pasteboard", "feature_featured_discontinued_personal", "feature_fewer_unbetrayed_drill", "feature_unhallowed_convulsionary_frenchification", "feature_choicest_ophthalmological_middlebrow", "feature_cornered_statutory_anglican", "feature_skint_maternal_carina", "feature_frothiest_sedged_summary", "feature_extended_cosier_smile", "feature_friesian_presentimental_cymbal", "feature_ecclesiological_invaluable_comatulid", "feature_involutional_antiseptic_isomerization", "feature_antipruritic_pourable_mete", "feature_interrogable_inane_erk", "feature_consuetudinary_relivable_monad", "feature_perdu_obligational_waratah", "feature_hebephrenic_furrowed_allottee", "feature_ratite_degree_expansibility", "feature_touristic_contingent_zincography", "feature_valleculate_pluralism_perfumery", "feature_saddening_czarist_quasar", "feature_observant_administrant_note", "feature_furry_flagellatory_febricity", "feature_jangling_showery_sitter", "feature_abducted_euphonic_pipewort", "feature_cercarian_aligning_soda", "feature_metempirical_sprawled_discontinuance", "feature_spoken_fractional_undset", "feature_jaggy_barer_responsum", "feature_cenozoic_bessarabian_kelt", "feature_stripiest_edged_sear", "feature_immunosuppressive_purgative_reformer", "feature_overcareful_infracostal_gallipoli", "feature_excommunicate_disturbed_mutule", "feature_penological_starting_nystatin", "feature_ligular_contemplative_laud", "feature_protochordate_connectible_futilitarian", "feature_contrite_evaporative_preformation", "feature_subtractive_emulsified_approving", "feature_confabulatory_malarian_phenotype", "feature_cliquish_unattached_gulbenkian", "feature_putrefiable_incommutable_citizen", "feature_middle_hunkered_alexandrite", "feature_obtainable_baddish_tuchun", "feature_sprightlier_albitic_justinian", "feature_decked_devilish_balladry", "feature_devolution_canty_suburb", "feature_mint_bilabial_redevelopment", "feature_erethistic_checkered_censurer", "feature_verdant_contrapuntal_urbanization", "feature_unoverthrown_unlined_exterminator", "feature_ultraviolet_sabbatical_galvanizer", "feature_mawkish_podgiest_venation", "feature_thixotropic_janiform_ilan", "feature_intercalary_shameful_carrefour", "feature_smorzando_conceited_dysphagia", "feature_lienteric_tricksy_aston", "feature_persian_expedite_cocky", "feature_pillowy_impelled_razzle", "feature_fourieristic_ecuadorian_pilotage", "feature_frumpiest_contusive_veteran", "feature_jonsonian_analyzable_carbamate", "feature_monastical_zoographical_sere", "feature_mercenary_pinchpenny_scrophularia", "feature_unspecialized_preclusive_lote", "feature_lacier_necrophilic_personage", "feature_neuromuscular_brutelike_ophite", "feature_advance_invalidated_marge", "feature_armed_keratose_slush", "feature_tiddly_divorcive_shoddy", "feature_equipotential_droopiest_molinism", "feature_observing_vigesimal_completion", "feature_cholagogue_reserved_silly", "feature_sensualistic_barbellate_moonstone", "feature_incredible_sipunculid_midriff", "feature_porkiest_waspy_recycling", "feature_oligochaete_grumpiest_cryptograph", "feature_metalloid_renascent_ferronickel", "feature_jehovist_kinglier_foxhole", "feature_cheerful_penicillate_plaza", "feature_saussuritic_unpurchased_provender", "feature_defoliated_called_lucubrator", "feature_lathiest_oblong_newton", "feature_lento_unborne_ethnomusicology", "feature_direct_expropriated_harping", "feature_unaneled_protractile_reviviscence", "feature_gainful_flighty_swampland", "feature_wearable_phoenician_congratulation", "feature_sickish_interlocutory_profligate", "feature_anarchic_ungual_planisphere", "feature_referential_eath_reconciliation", "feature_paradisal_predestinarian_bungler", "feature_xeric_chunkiest_homager", "feature_terminist_precocial_septarium", "feature_imaginative_monarchical_shive", "feature_trimonthly_appressed_siouan", "feature_recuperative_superscript_eunuchoidism", "feature_earthier_adjacent_hydropathy", "feature_immunosuppressive_pulmonate_asynergy", "feature_fortified_gasometrical_soccer", "feature_kookier_northward_disproof", "feature_harnessed_gratulant_nag", "feature_unsaluted_aloof_receiver", "feature_androgenic_monaxial_boarhound", "feature_asclepiadean_tenfold_bartender", "feature_adipose_diverging_analphabetic", "feature_podgy_wannest_protanomaly", "feature_lepidote_malevolent_maori", "feature_ungenteel_phanerozoic_grasmere", "feature_cured_holy_sporogonium", "feature_decillionth_stupefactive_bolshevism", "feature_quadrilingual_repayable_reconcilement", "feature_fake_participating_billionaire", "feature_bleak_clubbable_sodomy", "feature_wolfish_laic_canzone", "feature_loudish_molten_micher", "feature_procedural_approximal_centimeter", "feature_chorographic_laureate_dorsiflexion", "feature_unstitched_unsublimated_indelicacy", "feature_requitable_genuine_rule", "feature_chauvinistic_irksome_colloquy", "feature_unreclaimable_aggregative_meningioma", "feature_dimissory_gynandromorphic_manx", "feature_lackluster_thermic_bovid", "feature_exotic_socinian_stridence", "feature_conductible_indecomposable_athlete", "feature_synonymic_puckery_airhead", "feature_kentish_somnambulism_physiology", "feature_favorable_unincumbered_immortelle", "feature_museful_swinging_contactor", "feature_thriftiest_arriving_carucate", "feature_overloaded_tourist_lizzy", "feature_polyphonic_inconvenient_cointreau", "feature_unvaluable_falsest_vigil", "feature_jurisdictional_fermentative_contadino", "feature_appraising_unpayable_spiculum", "feature_falernian_unashamed_corroboration", "feature_nonconforming_tidal_hug", "feature_covered_mercurial_mariologist", "feature_wheeled_receptive_abstinent", "feature_unlockable_fornicate_lima", "feature_ratite_reverberatory_hooter", "feature_subtractive_randomized_exterritoriality", "feature_mozambican_genty_glimpse", "feature_progressional_sloshy_penology", "feature_lustred_jumbo_saphena", "feature_associable_hypersthenic_celibacy", "feature_astatic_sensitive_munshi", "feature_polar_splattered_analyser", "feature_chosen_grouty_broccoli", "feature_schlock_hypochondriacal_mezereon", "feature_operative_shellier_accompanist", "feature_sunburst_coeval_ceilidh", "feature_inopportune_episcopalian_seismism", "feature_unsoft_indwelling_kinema", "feature_haemolytic_finical_commentary", "feature_prolusory_vitrescible_solitudinarian", "feature_academic_turned_promenade", "feature_lethargic_grammatical_decathlon", "feature_bunchier_worsening_bracer", "feature_maestoso_peloponnesian_venue", "feature_sweeping_unconforming_cedar", "feature_brambly_jauntiest_pernancy", "feature_tardier_hillier_cradling", "feature_trollopy_bannered_vicomtesse", "feature_livable_gnotobiotic_inkblot", "feature_abrogative_hurt_lenition", "feature_exsert_whippy_calypso", "feature_undersexed_hedonic_spew", "feature_naked_cranial_cableway", "feature_monographical_paralytic_maroon", "feature_heartfelt_laddish_cuyp", "feature_uncleanly_circumgyratory_santonin", "feature_nearest_dawdling_nightingale", "feature_particularistic_unimpeached_figurant", "feature_formalized_smuttier_cottage", "feature_invected_dratted_garrick", "feature_abactinal_inventable_luminescence", "feature_taped_apeak_melodramatic", "feature_funky_amused_poppa", "feature_paramount_unheralded_ban", "feature_deposed_toughish_bribery", "feature_pilot_unshifting_cryptogamist", "feature_airy_divaricate_allomorph", "feature_forky_variolitic_impingement", "feature_gawkiest_shipboard_favouritism", "feature_alterable_tinted_kerosine", "feature_venetian_heating_tissot", "feature_maledictive_admired_dissimilation", "feature_caruncular_leafed_somnolency", "feature_overrank_unpanelled_puseyism", "feature_grizzliest_draughtier_shirting", "feature_figuline_sphincterial_palynology", "feature_analog_dozen_swami", "feature_utmost_assessorial_ayr", "feature_unskilled_sporangial_spock", "feature_tagalog_diverticular_soke", "feature_contractive_placental_foxhound", "feature_unpruned_allotriomorphic_finback", "feature_instant_compassionate_frump", "feature_unsocketed_autarkical_griselda", "feature_peptizing_machinable_computation", "feature_undistributed_problematic_agape", "feature_southernmost_necromantic_rental", "feature_paravail_urbanistic_adenosine", "feature_distractible_unreposing_arrowhead", "feature_hippiatric_tinctorial_slowpoke", "feature_dresden_uphill_thaumaturge", "feature_swift_tagalog_lacker", "feature_primary_patricidal_whitethorn", "feature_unhorsed_morphogenetic_affusion", "feature_tricentennial_subarcuate_ascendance", "feature_sycophantic_unrefined_calvinist", "feature_untressed_bicuspid_photograph", "feature_emergency_peckish_coequal", "feature_calorific_seigneurial_dietitian", "feature_isoelectric_comestible_chieftainship", "feature_supernatural_germicidal_detector", "feature_cobwebby_albescent_ophthalmoscope", "feature_peppier_jejune_dasyure", "feature_gamy_wattle_mescaline", "feature_potamic_indented_badalona", "feature_ultraviolet_willful_umbra", "feature_enceinte_directional_leishmania", "feature_eighty_lyric_hydrogen", "feature_vizierial_prevenient_component", "feature_chunky_fallen_erasure", "feature_princely_rabic_houri", "feature_premillennial_furuncular_founding", "feature_impressionable_untunable_macrocephaly", "feature_unscriptural_coconut_trisulphide", "feature_pyromantic_retaliative_internal", "feature_toed_accusatory_zoologist", "feature_sparing_outermost_sand", "feature_baleful_comfy_rubdown", "feature_majorcan_won_nicole", "feature_community_premandibular_fervor", "feature_carinate_mutational_incisor", "feature_winking_phlegmier_intro", "feature_subatomic_raffish_hexagram", "feature_road_coplanar_popsy", "feature_sensitive_inhaling_salting", "feature_climbable_terminative_lackluster", "feature_dulled_ablush_molybdenite", "feature_disaffected_formulism_tabbouleh", "feature_circumscribed_ratty_elma", "feature_jejune_statist_alerting", "feature_clastic_lacertilian_bothy", "feature_trustworthy_shouting_comorin", "feature_square_intrinsic_holi", "feature_archaean_port_respect", "feature_faded_breasted_additament", "feature_noncontroversial_predicted_snaggletooth", "feature_ferroelectric_rose_bootmaker", "feature_uninformative_kacha_wiggler", "feature_prevenient_lordlier_koftgar", "feature_sculpturesque_friended_meteorograph", "feature_aerobiotic_pickwickian_clergyman", "feature_doty_balletic_iona", "feature_unprotested_euphoric_engram", "feature_leachiest_plastery_arrayal", "feature_subtriangular_haughtiest_blunderer", "feature_histie_undeified_applicability", "feature_roadworthy_unbidden_asteroid", "feature_proxy_transcriptive_scaler", "feature_unvaccinated_cancroid_mentality", "feature_standardized_unbetrayed_noon", "feature_mentholated_acerose_academia", "feature_unintegrated_fore_hosteller", "feature_armorial_exclusory_forb", "feature_frisky_transuranic_spire", "feature_epithetic_diametrical_siphonage", "feature_cancelled_mickle_tubule", "feature_tripinnate_appropriate_size", "feature_cytoid_colonialism_brian", "feature_undeliverable_chorioid_ondine", "feature_geomedical_dendroidal_mismanagement", "feature_distal_indented_modernity", "feature_coky_humble_antecessor", "feature_unmade_mythological_orgeat", "feature_coarctate_hypnoid_hirudinean", "feature_ophthalmoscopic_biometric_univalence", "feature_platy_idiotic_vladimir", "feature_frowzy_transvestic_tympanum", "feature_orthopedical_siliculose_forger", "feature_predatory_cirriped_sapropel", "feature_contrate_interfacial_digestion", "feature_helminthoid_catalytical_tattle", "feature_rabbinical_undivorced_charr", "feature_germinant_sung_ketene", "feature_approximal_projectional_wrangler", "feature_sigmate_allergenic_eleven", "feature_toxophilitic_recidivism_bursar", "feature_aesthetic_stereographic_punjab", "feature_thermoluminescent_caboched_dishonor", "feature_haunched_stretchy_chicle", "feature_unpitied_jingoist_pyretology", "feature_austral_intrepid_sonia", "feature_overfull_negro_prurigo", "feature_chiselled_vasodilator_chiefdom", "feature_peritonitic_decadent_board", "feature_fell_unaligned_anesthetization", "feature_laziest_saronic_hornbeam", "feature_unconfinable_snuffly_cupid", "feature_elmier_unidentifiable_broccoli", "feature_liberated_lopsided_sixteenmo", "feature_uncleared_violable_arborvitae", "feature_unviable_anxiolytic_pyrene", "feature_campestral_tigerish_durrie", "feature_undefied_senary_siding", "feature_unfossilized_bankrupt_cannock", "feature_disclosed_mnemonic_ineffaceability", "feature_suspended_intracranial_fischer", "feature_shimmering_coverable_congolese", "feature_biserial_fulfilled_harpoon", "feature_pitiable_authoritative_clangor", "feature_abdominal_subtriplicate_fin", "feature_centenarian_ileac_caschrom", "feature_expected_beatified_coparcenary", "feature_unread_isopodan_ethic", "feature_china_fistular_phenylketonuria"]}


class GroupStatsPreProcessor(BaseProcessor):
    """
    WARNING: Only supported for v4.2 (Rain) data. The Rain dataset (re)introduced feature groups. \n
    
    Calculates group statistics for all data groups. \n
    :param groups: Groups to create features for. All groups by default. \n
    """
    def __init__(self, groups: list = None):
        super().__init__()
        self.all_groups = [
            'intelligence', 
            'charisma', 
            'strength', 
            'dexterity', 
            'constitution', 
            'wisdom', 
            'agility', 
            'serenity', 
            'sunshine', 
            'rain'
        ]
        self.group_names = groups if groups else self.all_groups
        self.feature_group_mapping = V4_2_FEATURE_GROUP_MAPPING

    @display_processor_info
    def transform(self, dataf: pd.DataFrame, *args, **kwargs) -> NumerFrame:
        """Check validity and add group features."""
        dataf = dataf.pipe(self._add_group_features)
        return NumerFrame(dataf)

    def _add_group_features(self, dataf: pd.DataFrame) -> pd.DataFrame:
        """Mean, standard deviation and skew for each group."""
        dataf = dataf.copy()
        for group in self.group_names:
            cols = self.feature_group_mapping[group]
            dataf.loc[:, f"feature_{group}_mean"] = dataf[cols].mean(axis=1)
            dataf.loc[:, f"feature_{group}_std"] = dataf[cols].std(axis=1)
            dataf.loc[:, f"feature_{group}_skew"] = dataf[cols].skew(axis=1)
            
        return dataf


class KatsuFeatureGenerator(BaseProcessor):
    """
    Effective feature engineering setup based on Katsu's starter notebook.
    Based on source by Katsu1110: https://www.kaggle.com/code1110/numeraisignals-starter-for-beginners

    :param windows: Time interval to apply for window features: \n
    1. Percentage Rate of change \n
    2. Volatility \n
    3. Moving Average gap \n
    :param ticker_col: Columns with tickers to iterate over. \n
    :param close_col: Column name where you have closing price stored.
    """

    warnings.filterwarnings("ignore")

    def __init__(
        self,
        windows: list,
        ticker_col: str = "ticker",
        close_col: str = "close",
        num_cores: int = None,
    ):
        super().__init__()
        self.windows = windows
        self.ticker_col = ticker_col
        self.close_col = close_col
        self.num_cores = num_cores if num_cores else os.cpu_count()

    @display_processor_info
    def transform(self, dataf: Union[pd.DataFrame, NumerFrame]) -> NumerFrame:
        """Multiprocessing feature engineering."""
        tickers = dataf.loc[:, self.ticker_col].unique().tolist()
        rich_print(
            f"Feature engineering for {len(tickers)} tickers using {self.num_cores} CPU cores."
        )
        dataf_list = [
            x
            for _, x in tqdm(
                dataf.groupby(self.ticker_col), desc="Generating ticker DataFrames"
            )
        ]
        dataf = self._generate_features(dataf_list=dataf_list)
        return NumerFrame(dataf)

    def feature_engineering(self, dataf: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering for single ticker."""
        close_series = dataf.loc[:, self.close_col]
        for x in self.windows:
            dataf.loc[
                :, f"feature_{self.close_col}_ROCP_{x}"
            ] = close_series.pct_change(x)

            dataf.loc[:, f"feature_{self.close_col}_VOL_{x}"] = (
                np.log1p(close_series).pct_change().rolling(x).std()
            )

            dataf.loc[:, f"feature_{self.close_col}_MA_gap_{x}"] = (
                close_series / close_series.rolling(x).mean()
            )

        dataf.loc[:, "feature_RSI"] = self._rsi(close_series)
        macd, macd_signal = self._macd(close_series)
        dataf.loc[:, "feature_MACD"] = macd
        dataf.loc[:, "feature_MACD_signal"] = macd_signal
        return dataf.bfill()

    def _generate_features(self, dataf_list: list) -> pd.DataFrame:
        """Add features for list of ticker DataFrames and concatenate."""
        with Pool(self.num_cores) as p:
            feature_datafs = list(
                tqdm(
                    p.imap(self.feature_engineering, dataf_list),
                    desc="Generating features",
                    total=len(dataf_list),
                )
            )
        return pd.concat(feature_datafs)

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        See source https://github.com/peerchemist/finta
        and fix https://www.tradingview.com/wiki/Talk:Relative_Strength_Index_(RSI)
        """
        delta = close.diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        gain = up.ewm(com=(period - 1), min_periods=period).mean()
        loss = down.abs().ewm(com=(period - 1), min_periods=period).mean()

        rs = gain / loss
        return pd.Series(100 - (100 / (1 + rs)))

    def _macd(
        self, close: pd.Series, span1=12, span2=26, span3=9
    ) -> Tuple[pd.Series, pd.Series]:
        """Compute MACD and MACD signal."""
        exp1 = self.__ema1(close, span1)
        exp2 = self.__ema1(close, span2)
        macd = 100 * (exp1 - exp2) / exp2
        signal = self.__ema1(macd, span3)
        return macd, signal

    @staticmethod
    def __ema1(series: pd.Series, span: int) -> pd.Series:
        """Exponential moving average"""
        a = 2 / (span + 1)
        return series.ewm(alpha=a).mean()


class EraQuantileProcessor(BaseProcessor):
    """
    Transform features into quantiles on a per-era basis

    :param num_quantiles: Number of buckets to split data into. \n
    :param era_col: Era column name in the dataframe to perform each transformation. \n
    :param features: All features that you want quantized. All feature cols by default. \n
    :param num_cores: CPU cores to allocate for quantile transforming. All available cores by default. \n
    :param random_state: Seed for QuantileTransformer. \n
    :param batch_size: How many feature to process at the same time.
    For Numerai Signals scale data it is advisable to process features one by one. 
    This is the default setting.
    """

    def __init__(
        self,
        num_quantiles: int = 50,
        era_col: str = "friday_date",
        features: list = None,
        num_cores: int = None,
        random_state: int = 0,
        batch_size: int = 1
    ):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.era_col = era_col
        self.num_cores = num_cores if num_cores else os.cpu_count()
        self.features = features 
        self.random_state = random_state
        self.batch_size = batch_size 

    def _process_eras(self, groupby_object):
        quantizer = QuantileTransformer(
            n_quantiles=self.num_quantiles, random_state=self.random_state
        )
        qt = lambda x: quantizer.fit_transform(x.values.reshape(-1, 1)).ravel()

        column = groupby_object.transform(qt)
        return column

    @display_processor_info
    def transform(
        self,
        dataf: Union[pd.DataFrame, NumerFrame],
    ) -> NumerFrame:
        """Multiprocessing quantile transforms by era."""
        features = self.features if self.features else dataf.feature_cols
        rich_print(
            f"Quantiling for {len(features)} features using {self.num_cores} CPU cores."
        )

        date_groups = dataf.groupby(self.era_col)
        for batch_start in tqdm(range(0, len(features), self.batch_size), total=len(features)):
            # Create batch of features. Default is to process features on by one.
            batch_end = min(batch_start + self.batch_size, len(features))
            batch_features = features[batch_start:batch_end]
            groupby_objects = [date_groups[feature] for feature in batch_features]

            with Pool() as p:
                results = list(
                        p.imap(self._process_eras, groupby_objects),
                )

            quantiles = pd.concat(results, axis=1)
            dataf[
                [f"{feature}_quantile{self.num_quantiles}" for feature in batch_features]
            ] = quantiles
            return NumerFrame(dataf)


class TickerMapper(BaseProcessor):
    """
    Map ticker from one format to another. \n
    :param ticker_col: Column used for mapping. Must already be present in the input data. \n
    :param target_ticker_format: Format to map tickers to. Must be present in the ticker map. \n
    For default mapper supported ticker formats are: ['ticker', 'bloomberg_ticker', 'yahoo'] \n
    :param mapper_path: Path to CSV file containing at least ticker_col and target_ticker_format columns. \n
    Can be either a web link of local path. Numerai Signals mapping by default.
    """

    def __init__(
        self, ticker_col: str = "ticker", target_ticker_format: str = "bloomberg_ticker",
        mapper_path: str = "https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_ticker_map_w_bbg.csv"
    ):
        super().__init__()
        self.ticker_col = ticker_col
        self.target_ticker_format = target_ticker_format

        self.signals_map_path = mapper_path
        self.ticker_map = pd.read_csv(self.signals_map_path)

        assert (
            self.ticker_col in self.ticker_map.columns
        ), f"Ticker column '{self.ticker_col}' is not available in ticker mapping."
        assert (
            self.target_ticker_format in self.ticker_map.columns
        ), f"Target ticker column '{self.target_ticker_format}' is not available in ticker mapping."

        self.mapping = dict(
            self.ticker_map[[self.ticker_col, self.target_ticker_format]].values
        )

    @display_processor_info
    def transform(
        self, dataf: Union[pd.DataFrame, NumerFrame], *args, **kwargs
    ) -> NumerFrame:
        dataf[self.target_ticker_format] = dataf[self.ticker_col].map(self.mapping)
        return NumerFrame(dataf)

# %% ../nbs/03_preprocessing.ipynb 66
class SignalsTargetProcessor(BaseProcessor):
    """
    Engineer targets for Numerai Signals. \n
    More information on implements Numerai Signals targets: \n
    https://forum.numer.ai/t/decoding-the-signals-target/2501

    :param price_col: Column from which target will be derived. \n
    :param windows: Timeframes to use for engineering targets. 10 and 20-day by default. \n
    :param bins: Binning used to create group targets. Nomi binning by default. \n
    :param labels: Scaling for binned target. Must be same length as resulting bins (bins-1). Numerai labels by default.
    """

    def __init__(
        self,
        price_col: str = "close",
        windows: list = None,
        bins: list = None,
        labels: list = None,
    ):
        super().__init__()
        self.price_col = price_col
        self.windows = windows if windows else [10, 20]
        self.bins = bins if bins else [0, 0.05, 0.25, 0.75, 0.95, 1]
        self.labels = labels if labels else [0, 0.25, 0.50, 0.75, 1]

    @display_processor_info
    def transform(self, dataf: NumerFrame) -> NumerFrame:
        for window in tqdm(self.windows, desc="Signals target engineering windows"):
            dataf.loc[:, f"target_{window}d_raw"] = (
                dataf[self.price_col].pct_change(periods=window).shift(-window)
            )
            era_groups = dataf.groupby(dataf.meta.era_col)

            dataf.loc[:, f"target_{window}d_rank"] = era_groups[
                f"target_{window}d_raw"
            ].rank(pct=True, method="first")
            dataf.loc[:, f"target_{window}d_group"] = era_groups[
                f"target_{window}d_rank"
            ].transform(
                lambda group: pd.cut(
                    group, bins=self.bins, labels=self.labels, include_lowest=True
                )
            )
        return NumerFrame(dataf)


class LagPreProcessor(BaseProcessor):
    """
    Add lag features based on given windows.

    :param windows: All lag windows to process for all features. \n
    [5, 10, 15, 20] by default (4 weeks lookback) \n
    :param ticker_col: Column name for grouping by tickers. \n
    :param feature_names: All features for which you want to create lags. All features by default.
    """

    def __init__(
        self,
        windows: list = None,
        ticker_col: str = "bloomberg_ticker",
        feature_names: list = None,
    ):
        super().__init__()
        self.windows = windows if windows else [5, 10, 15, 20]
        self.ticker_col = ticker_col
        self.feature_names = feature_names

    @display_processor_info
    def transform(self, dataf: NumerFrame, *args, **kwargs) -> NumerFrame:
        feature_names = self.feature_names if self.feature_names else dataf.feature_cols
        ticker_groups = dataf.groupby(self.ticker_col)
        for feature in tqdm(feature_names, desc="Lag feature generation"):
            feature_group = ticker_groups[feature]
            for day in self.windows:
                shifted = feature_group.shift(day, axis=0)
                dataf.loc[:, f"{feature}_lag{day}"] = shifted
        return NumerFrame(dataf)


class DifferencePreProcessor(BaseProcessor):
    """
    Add difference features based on given windows. Run LagPreProcessor first.

    :param windows: All lag windows to process for all features. \n
    :param feature_names: All features for which you want to create differences. All features that also have lags by default. \n
    :param pct_change: Method to calculate differences. If True, will calculate differences with a percentage change. Otherwise calculates a simple difference. Defaults to False \n
    :param abs_diff: Whether to also calculate the absolute value of all differences. Defaults to True \n
    """

    def __init__(
        self,
        windows: list = None,
        feature_names: list = None,
        pct_diff: bool = False,
        abs_diff: bool = False,
    ):
        super().__init__()
        self.windows = windows if windows else [5, 10, 15, 20]
        self.feature_names = feature_names
        self.pct_diff = pct_diff
        self.abs_diff = abs_diff

    @display_processor_info
    def transform(self, dataf: NumerFrame, *args, **kwargs) -> NumerFrame:
        feature_names = self.feature_names if self.feature_names else dataf.feature_cols
        for feature in tqdm(feature_names, desc="Difference feature generation"):
            lag_columns = dataf.get_pattern_data(f"{feature}_lag").columns
            if not lag_columns.empty:
                for day in self.windows:
                    differenced_values = (
                        (dataf[feature] / dataf[f"{feature}_lag{day}"]) - 1
                        if self.pct_diff
                        else dataf[feature] - dataf[f"{feature}_lag{day}"]
                    )
                    dataf[f"{feature}_diff{day}"] = differenced_values
                    if self.abs_diff:
                        dataf[f"{feature}_absdiff{day}"] = np.abs(
                            dataf[f"{feature}_diff{day}"]
                        )
            else:
                rich_print(
                    f":warning: WARNING: Skipping {feature}. Lag features for feature: {feature} were not detected. Have you already run LagPreProcessor? :warning:"
                )
        return NumerFrame(dataf)


class PandasTaFeatureGenerator:
    """
    Generate features with pandas-ta.
    https://github.com/twopirllc/pandas-ta

    :param strategy: Valid Pandas Ta strategy. \n
    For more information on creating a strategy, see: \n
    https://github.com/twopirllc/pandas-ta#pandas-ta-strategy \n
    By default, a strategy with RSI(14) and RSI(60) is used. \n
    :param ticker_col: Column name for grouping by tickers. \n
    :param num_cores: Number of cores to use for multiprocessing. \n
    By default, all available cores are used. \n
    """
    def __init__(self, 
                 strategy: ta.Strategy = None,
                 ticker_col: str = "ticker",
                 num_cores: int = None,
    ):
        super().__init__()
        self.ticker_col = ticker_col
        self.num_cores = num_cores if num_cores else os.cpu_count()
        standard_strategy = ta.Strategy(name="standard", 
                                        ta=[{"kind": "rsi", "length": 14, "col_names": ("feature_RSI_14")},
                                            {"kind": "rsi", "length": 60, "col_names": ("feature_RSI_60")}])
        self.strategy = strategy if strategy is not None else standard_strategy

    @display_processor_info
    def transform(self, dataf: Union[pd.DataFrame, NumerFrame]) -> NumerFrame:
        """
        Main feature generation method. \n 
        :param dataf: DataFrame with columns: [ticker, date, open, high, low, close, volume] \n
        :return: DataFrame with features added.
        """
        dataf_list = [
            x
            for _, x in tqdm(
                dataf.groupby(self.ticker_col), desc="Generating ticker DataFrames"
            )
        ]
        dataf = self._generate_features(dataf_list=dataf_list)
        return NumerFrame(dataf)
    
    def _generate_features(self, dataf_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Add features for list of ticker DataFrames and concatenate.
        :param dataf_list: List of DataFrames for each ticker.
        :return: Concatenated DataFrame for all full list with features added.
        """
        with Pool(self.num_cores) as p:
            feature_datafs = list(
                tqdm(
                    p.imap(self.add_features, dataf_list),
                    desc="Generating pandas-ta features",
                    total=len(dataf_list),
                )
            )
        return pd.concat(feature_datafs)

    def add_features(self, ticker_df: pd.DataFrame) -> pd.DataFrame:
        """ 
        The TA strategy is applied to the DataFrame here.
        :param ticker_df: DataFrame for a single ticker.
        :return: DataFrame with features added.
        """
        # We use a different multiprocessing engine so shutting off pandas_ta's multiprocessing
        ticker_df.ta.cores = 0
        ticker_df.ta.strategy(self.strategy)
        return ticker_df


class AwesomePreProcessor(BaseProcessor):
    """ TEMPLATE - Do some awesome preprocessing. """
    def __init__(self):
        super().__init__()

    @display_processor_info
    def transform(self, dataf: NumerFrame, *args, **kwargs) -> NumerFrame:
        # Do processing
        ...
        # Parse all contents of NumerFrame to the next pipeline step
        return NumerFrame(dataf)
