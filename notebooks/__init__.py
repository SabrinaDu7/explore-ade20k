import importlib

_analysis = importlib.import_module("notebooks.03_analysis")
_overview = importlib.import_module("notebooks.01_overview")
_samples = importlib.import_module("notebooks.02_samples")

# from 03_analysis
compute_class_area = _analysis.compute_class_area
build_image_class_dataframe = _analysis.build_image_class_dataframe
class_area_range = _analysis.class_area_range
compute_object_count = _analysis.compute_object_count
class_stats_dataframe = _analysis.class_stats_dataframe
plot_area_distribution = _analysis.plot_area_distribution
plot_metric_histogram = _analysis.plot_metric_histogram
plot_size_vs_instances_violin = _analysis.plot_size_vs_instances_violin
filename_idx_to_img_idx = _analysis.filename_idx_to_img_idx
img_idx_to_filename_idx = _analysis.img_idx_to_filename_idx

# from 01_overview
absent_classes = _overview.absent_classes
plot_label_distribution = _overview.plot_label_distribution
plot_resolution_distribution = _overview.plot_resolution_distribution
plot_aspect_ratio_distribution = _overview.plot_aspect_ratio_distribution
plot_resolution_vs_aspect = _overview.plot_resolution_vs_aspect

# from 02_samples
show_val_samples = _samples.show_val_samples
