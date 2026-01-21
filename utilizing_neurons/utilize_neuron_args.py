import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

from utils.utils import setup_logger, str2bool, ArgumentSaver, AddDefaultInformationAction, float_or_float_tuple_type, int_or_int_tuple_type

def get_utilize_neuron_args():
    saver = ArgumentSaver()

    # general
    saver.add_argument('--random_seed', default=None, type=int)
    saver.add_argument('--count_epochs', type=int, default=10, action=AddDefaultInformationAction)
    saver.add_argument('--batch_size', type=int, default=32, action=AddDefaultInformationAction)
    saver.add_argument('--enable_progress_bar', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--enable_plotting', type=str2bool, nargs='?', const=True, default=True)
    saver.add_argument('--plot_train_every_in_epochs', type=int, default=5)
    saver.add_argument('--plot_valid_every_in_epochs', type=int, default=5)
    saver.add_argument('--plot_train_every_in_batches', type=int, default=10)
    saver.add_argument('--plot_valid_every_in_batches', type=int, default=10)
    saver.add_argument('--enable_checkpointing', type=str2bool, nargs='?', const=True, default=True)
    saver.add_argument('--checkpoint_every_in_epochs', type=int, default=20)
    saver.add_argument('--checkpoint_first_k_epochs', type=int, default=20)
    saver.add_argument('--checkpoint_top_k_valid_accuracies', type=int, default=5)
    saver.add_argument('--checkpoint_top_k_valid_aucs', type=int, default=0)
    saver.add_argument('--checkpoint_bottom_k_valid_maes', type=int, default=0)
    saver.add_argument('--utilizer_from_checkpoint', default=None)
    saver.add_argument('--only_calculate_metrics', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--calculate_metrics_save_to_folder', type=str2bool, nargs='?', const=True, default=False)

    # ds selection - general options
    saver.add_argument('--odd_labels_vs_even_labels', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--ssc_odd_number_labels_vs_all_other_labels', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--label_x_vs_all', default=None, type=int)
    saver.add_argument('--label_x_vs_all_label_x_weight', default=1, type=float)
    saver.add_argument('--keep_labels', default=None)
    saver.add_argument('--one_hot_size', default=None, type=int)
    saver.add_argument('--split_seed', default=None, type=int)
    saver.add_argument('--valid_percentage', default=0.2, type=float)
    saver.add_argument('--max_count_samples', default=None, type=int)

    # ds selection - spike ds options
    saver.add_argument('--stimulus_duration_in_ms', default=420, type=int)
    saver.add_argument('--average_stimulus_firing_rate_per_axon', default=20, type=int) # TODO: float?
    saver.add_argument('--average_stimulus_burst_firing_rate_per_axon', default=None, type=int) # TODO: float?
    saver.add_argument('--average_background_firing_rate_per_axon', default=0, type=float)
    saver.add_argument('--spike_jitter', default=2.5, type=float)
    saver.add_argument('--background_firing_rate_duration_from_start', default=-1, type=int) # -1 means args.stimulus_duration_in_ms
    saver.add_argument('--count_exc_axons', default=800, type=int)
    saver.add_argument('--count_exc_bias_axons', default=None, type=int) # part of count_exc_axons
    saver.add_argument('--count_inh_axons', default=200, type=int)
    saver.add_argument('--count_inh_bias_axons', default=None, type=int) # part of count_exc_axons
    saver.add_argument('--gabor_method', default=None, type=int)
    saver.add_argument('--gabor_c1_only_unit_0', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--subsample_modulo', default=None, type=int)
    saver.add_argument('--binarization_method', default=None, type=int)
    saver.add_argument('--initial_image_size', default=None, type=int, short_name='iis')
    saver.add_argument('--presampled', type=str2bool, nargs='?', const=True, default=True)
    saver.add_argument('--reduce_fr', type=str2bool, nargs='?', const=True, default=False, short_name="rfr")
    saver.add_argument('--temporal_adaptation', type=str2bool, nargs='?', const=True, default=False, short_name="ta")
    saver.add_argument('--arange_spikes', type=str2bool, nargs='?', const=True, default=False, short_name="arsp")
    saver.add_argument('--temporal_delay_from_start', default=10, type=int, short_name="tD")
    saver.add_argument('--detect_onsets_offsets', type=str2bool, nargs='?', const=True, default=False, short_name="doo")
    saver.add_argument('--detect_onsets_offsets_window_size', default=None, type=int, short_name="doow")
    saver.add_argument('--detect_onsets_offsets_threshold', default=None, type=float, short_name="dooth")
    saver.add_argument('--detect_onsets_offsets_sustained', type=str2bool, nargs='?', const=True, default=False, short_name="doos")
    saver.add_argument('--detect_onsets_offsets_sustained_window_size', default=None, type=int, short_name="doosw")
    saver.add_argument('--detect_onsets_offsets_sustained_overlap', default=None, type=int, short_name="doosol")
    saver.add_argument('--detect_onsets_offsets_sustained_onset_threshold', default=None, type=float, short_name="doosoth")
    saver.add_argument('--detect_onsets_offsets_sustained_offset_threshold', default=None, type=float, short_name="doosoth")
    saver.add_argument('--envelope_extraction', type=str2bool, nargs='?', const=True, default=False, short_name="ee")
    saver.add_argument('--envelope_extraction_kernel_size', default=None, type=int, short_name="eeks")
    saver.add_argument('--envelope_extraction_threshold', default=None, type=float, short_name="eeth")
    saver.add_argument('--subsample_envelope', type=str2bool, nargs='?', const=True, default=False, short_name="see")
    saver.add_argument('--subsample_envelope_time_window', default=None, type=int, short_name="seetw")
    saver.add_argument('--subsample_envelope_axon_group', default=None, type=int, short_name="seeg")
    saver.add_argument('--subsample_envelope_statistic', default=None, type=int, short_name="sees")
    saver.add_argument('--binarise_subsample_envelope', type=str2bool, nargs='?', const=True, default=False, short_name="bse")
    saver.add_argument('--binarised_subsample_envelope_and_detect_onsets_offsets_sustained_not_binary', type=str2bool, nargs='?', const=True, default=False, short_name="bseandoosnb")
    saver.add_argument('--hierarchical_audio_processing', type=str2bool, nargs='?', const=True, default=False, short_name="hap")
    saver.add_argument('--hierarchical_audio_processing2', type=str2bool, nargs='?', const=True, default=False, short_name="hap2")
    saver.add_argument('--wav_to_binary_features', type=str2bool, nargs='?', const=True, default=False, short_name="wbf")
    saver.add_argument('--ds_shorter_name', type=str2bool, nargs='?', const=True, default=False, short_name="dsSn")
    saver.add_argument('--extra_label_information', type=str2bool, nargs='?', const=True, default=True, short_name="eli")

    # ds selection - spiking classification ds
    saver.add_argument('--abstract', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--spiking_abstract', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--abstract_count_dimensions', default=2, type=int, short_name="absD")
    saver.add_argument('--abstract_count_values_per_dimension', default=2, type=int, short_name="absN")
    # TODO: have a specific random_seed for abstract random variants
    saver.add_argument('--abstract_random_labels', type=str2bool, nargs='?', const=True, default=False, short_name="absRandLbl")
    saver.add_argument('--abstract_random_permutation_labels', type=str2bool, nargs='?', const=True, default=False, short_name="absRandPermLbl")
    saver.add_argument('--abstract_mutually_exclusive_synapses_for_each_pattern', type=str2bool, nargs='?', const=True, default=False, short_name="absMut") # TODO: global?
    saver.add_argument('--abstract_samples_per_pattern', default=50, type=int, short_name="samp") # TODO: global?
    saver.add_argument('--abstract_shuffle_inds', type=str2bool, nargs='?', const=True, default=True, short_name="shuf") # TODO: global?
    saver.add_argument('--abstract_num_t', default=None, type=int, short_name="absNumT") # TODO: global?
    saver.add_argument('--abstract_trial', default=None, type=int, short_name="absT") # TODO: global?
    saver.add_argument('--abstract_input_encoding', default='opt', short_name="absE") # TODO: global?
    saver.add_argument('--abstract_use_raw_patterns', type=str2bool, nargs='?', const=True, default=False) # TODO: global?
    saver.add_argument('--abstract_jitter', default=None, type=float, short_name="absJ") # TODO: global?
    saver.add_argument('--abstract_valid_jitter', default=None, type=float, short_name="absVJ") # TODO: global?
    saver.add_argument('--abstract_valid_jitters_start', default=None, type=float, short_name="absVJS")
    saver.add_argument('--abstract_valid_jitters_end', default=None, type=float, short_name="absVJE")
    saver.add_argument('--abstract_valid_jitters_step', default=None, type=float, short_name="absVJSt")

    saver.add_argument('--spiking_cat_and_dog', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--non_spiking_spiking_cat_and_dog', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--spiking_afhq', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--non_spiking_spiking_afhq', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--shd', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--non_spiking_shd', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--ssc', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--non_spiking_ssc', type=str2bool, nargs='?', const=True, default=False)

    # model selection - baselines
    saver.add_argument('--logistic_regression', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--logistic_regression_bias', type=str2bool, nargs='?', const=True, default=True)
    saver.add_argument('--spiking_logistic_regression', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--spiking_logistic_regression_beta', default=1, type=float)
    saver.add_argument('--sklearn_logistic_regression', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--sklearn_svm', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--sklearn_mlp', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--sklearn_mlp_hidden_layer_size', default=256, type=int)
    saver.add_argument('--fully_connected', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--fully_connected_bias', type=str2bool, nargs='?', const=True, default=True)
    saver.add_argument('--fully_connected_batch_norm', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--fully_connected_count_hidden_layers', default=1, type=int)
    saver.add_argument('--fully_connected_hidden_layer_size', default=128, type=int)

    # model selection - neurons
    saver.add_argument('--neuron_model_folder', default=None)
    saver.add_argument('--neuron_model_nseg', default=None, type=int)
    saver.add_argument('--neuron_model_max_segment_length', default=None, type=float)

    # model selection - neuron/model nn/tcn
    saver.add_argument('--neuron_nn_file', default=None, short_name='nnn')
    saver.add_argument('--neuron_nn_threshold', default=0.9, short_name='nnnTh', type=float)

    saver.add_argument('--model_from_checkpoint', default=None)

    # utilization arguments
    saver.add_argument('--time_left_padding_firing_rate', default=1.0, type=float)
    saver.add_argument('--time_left_padding_extra_in_ms', default=0, type=int)
    saver.add_argument('--time_left_padding_before_wiring', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--freeze_model', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--use_wiring_layer', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--positive_wiring', type=str2bool, nargs='?', const=True, default=True)
    saver.add_argument('--grad_abs', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--positive_by_sigmoid', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--positive_by_softplus', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--wiring_zero_smaller_than', default=None, type=float_or_float_tuple_type, short_name='wirZsmlThn')
    saver.add_argument('--wiring_keep_max_k_from_input', default=None, type=int_or_int_tuple_type, short_name='wirKmaxRk')
    saver.add_argument('--wiring_keep_max_k_to_output', default=None, type=int_or_int_tuple_type, short_name='wirKmaxCk')
    saver.add_argument('--wiring_keep_weight_mean', default=None, type=float_or_float_tuple_type, short_name='wirKmean')
    saver.add_argument('--wiring_keep_weight_std', default=None, type=float_or_float_tuple_type, short_name='wirKstd')
    saver.add_argument('--wiring_keep_weight_max', default=None, type=float_or_float_tuple_type, short_name='wirKmax')
    saver.add_argument('--wiring_enforce_every_in_train_epochs', default=None, type=int_or_int_tuple_type, short_name='wirEnfTe')
    saver.add_argument('--wiring_enforce_every_in_train_batches', default=None, type=int_or_int_tuple_type, short_name='wirEnfTb')
    saver.add_argument('--wiring_dales_law', type=str2bool, nargs='?', const=True, default=False, short_name='wirDle')
    saver.add_argument('--wiring_bias', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--wiring_weight_init_mean', default=0.0, type=float_or_float_tuple_type, short_name='wirmean')
    saver.add_argument('--wiring_weight_init_bound', default=None, type=float_or_float_tuple_type, short_name='wirbnd')
    saver.add_argument('--wiring_weight_init_sparsity', default=None, type=float_or_float_tuple_type, short_name='wirsprs')
    saver.add_argument('--wiring_weight_l1_reg', default=0.0, type=float_or_float_tuple_type, short_name='wirl1')
    saver.add_argument('--wiring_weight_l2_reg', default=0.0, type=float_or_float_tuple_type, short_name='wirl2')
    saver.add_argument('--functional_only_wiring', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--population_k', type=int, default=1, action=AddDefaultInformationAction)
    saver.add_argument('--use_population_masking_layer', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--population_masking_bias', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--population_masking_weight_init_mean', default=0.0, type=float)
    saver.add_argument('--population_masking_weight_init_bound', default=None, type=float)
    saver.add_argument('--population_masking_weight_init_sparsity', default=None, type=float)
    saver.add_argument('--functional_only_population_masking', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--decoding_type', default='max_pooling', action=AddDefaultInformationAction)
    saver.add_argument('--decoding_time_from_end', default=None, type=int)
    saver.add_argument('--require_no_spikes_before_decoding_time', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--disable_model_last_layer', type=str2bool, nargs='?', const=True, default=False)    
    saver.add_argument('--optimizer', default='adam', action=AddDefaultInformationAction)
    saver.add_argument('--lr', default=1e-3, type=float)
    saver.add_argument('--step_lr', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--momentum', default=0.9, type=float)
    saver.add_argument('--weight_decay', default=5e-4, type=float)
    saver.add_argument('--differentiable_binarization_threshold_surrogate_spike', type=str2bool, nargs='?', const=True, default=True)
    saver.add_argument('--differentiable_binarization_threshold_surrogate_spike_beta', default=5, type=int)
    saver.add_argument('--differentiable_binarization_threshold_straight_through', type=str2bool, nargs='?', const=True, default=False)

    return saver