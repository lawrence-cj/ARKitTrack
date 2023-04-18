from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.arkittrack_path = '/data2/iRGBD/vot_test'
    settings.cdtb_path = '/data1/Datasets/VOT/RGBD19'
    settings.davis_dir = ''
    settings.depthtrack_path = '/data2/Datasets/DepthTrack/DepthTrackTest'

    settings.got10k_lmdb_path = '/data2/Desktop/ART_VOT/data/got10k_lmdb'
    settings.got10k_path = '/data2/Desktop/ART_VOT/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/data2/Desktop/ART_VOT/data/itb'
    settings.lasot_extension_subset_path_path = '/data2/Desktop/ART_VOT/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/data2/Desktop/ART_VOT/data/lasot_lmdb'
    settings.lasot_path = '/data2/Desktop/ART_VOT/data/lasot'
    settings.network_path = '/data2/Desktop/ART_VOT/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/data2/Desktop/ART_VOT/data/nfs'
    settings.otb_path = '/data2/Desktop/ART_VOT/data/otb'
    settings.prj_dir = '/data2/Desktop/ART_VOT'
    settings.result_plot_path = '/data2/Desktop/ART_VOT/output/test/result_plots'
    settings.results_path = '/data2/Desktop/ART_VOT/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data2/Desktop/ART_VOT/output'
    settings.segmentation_path = '/data2/Desktop/ART_VOT/output/test/segmentation_results'
    settings.tc128_path = '/data2/Desktop/ART_VOT/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/data2/Desktop/ART_VOT/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data2/Desktop/ART_VOT/data/trackingnet'
    settings.uav_path = '/data2/Desktop/ART_VOT/data/uav'
    settings.vot18_path = '/data2/Desktop/ART_VOT/data/vot2018'
    settings.vot22_path = '/data2/Desktop/ART_VOT/data/vot2022'
    settings.vot_path = '/data2/Desktop/ART_VOT/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

