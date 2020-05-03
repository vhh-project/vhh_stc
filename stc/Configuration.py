from stc.utils import *
import yaml


class Configuration:
    def __init__(self, config_file: str):
        printCustom("create instance of configuration ... ", STDOUT_TYPE.INFO)

        self.config_file = config_file;

        self.debug_flag = -1
        self.sbd_results_path = None
        self.save_debug_pkg_flag = -1

        self.resize_dim = None;
        self.flag_convert2Gray = -1;
        self.flag_crop = -1;
        self.flag_downscale = -1;
        self.opt_histogram_equ = None;

        self.activate_candidate_selection = -1;

        self.class_names = None
        self.batch_size = -1
        self.save_raw_results = -1;
        self.number_of_frames_per_shot = -1
        self.path_postfix_raw_results = None;
        self.path_prefix_raw_results = None;
        self.path_raw_results = None;

        self.save_final_results = -1;
        self.path_prefix_final_results = None;
        self.path_postfix_final_results = None;
        self.path_final_results = None;

        self.path_videos = None;
        self.threshold = 0.0;
        self.threshold_mode = None;
        self.window_size = None;
        self.backbone_cnn = None;
        self.similarity_metric = None;

        self.path_pre_trained_model = None;

        self.path_eval_results = None;
        self.path_raw_results_eval = None;
        self.save_eval_results = -1;
        self.path_gt_data = None;

    def loadConfig(self):
        fp = open(self.config_file, 'r');
        config = yaml.load(fp, Loader=yaml.BaseLoader);

        developer_config = config['Development'];
        pre_processing_config = config['PreProcessing'];
        stc_core_config = config['StcCore'];
        evaluation_config = config['Evaluation'];

        # developer_config section
        self.debug_flag = int(developer_config['DEBUG_FLAG'])
        self.sbd_results_path = developer_config['SBD_RESULTS_PATH']
        self.save_debug_pkg_flag = int(developer_config['SAVE_DEBUG_PKG'])

        # pre-processing section
        self.resize_dim = (int(pre_processing_config['RESIZE_DIM'].split(',')[0]),
                           int(pre_processing_config['RESIZE_DIM'].split(',')[1]))

        # stc_core_config section
        self.class_names = stc_core_config['CLASS_NAMES']
        self.batch_size = int(stc_core_config['BATCH_SIZE'])

        self.save_raw_results = int(stc_core_config['SAVE_RAW_RESULTS'])
        self.number_of_frames_per_shot = int(stc_core_config['NUMBER_OF_FRAMES_PER_SHOT'])

        self.path_postfix_raw_results = stc_core_config['POSTFIX_RAW_RESULTS']
        self.path_prefix_raw_results = stc_core_config['PREFIX_RAW_RESULTS']
        self.path_raw_results = stc_core_config['PATH_RAW_RESULTS']

        self.save_final_results = int(stc_core_config['SAVE_FINAL_RESULTS'])
        self.path_prefix_final_results = stc_core_config['PREFIX_FINAL_RESULTS']
        self.path_postfix_final_results = stc_core_config['POSTFIX_FINAL_RESULTS']
        self.path_final_results = stc_core_config['PATH_FINAL_RESULTS'];

        self.path_videos = stc_core_config['PATH_VIDEOS'];
        self.threshold = float(stc_core_config['THRESHOLD']);
        self.path_pre_trained_model = stc_core_config['PATH_PRETRAINED_MODEL'];

        # evaluation section
        self.path_raw_results_eval = evaluation_config['PATH_RAW_RESULTS'];
        self.path_eval_results = evaluation_config['PATH_EVAL_RESULTS'];
        self.save_eval_results = int(evaluation_config['SAVE_EVAL_RESULTS']);
        self.path_gt_data = evaluation_config['PATH_GT_ANNOTATIONS'];





