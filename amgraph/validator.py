from .validators import EstimationValidator, EstimationValidator, SGDValidator


def main():
    # EstimationValidator.multi_run("combine_all_le30", max_num_iter=30, early_stop=True, only_val_once= False)
    # EstimationValidator.multi_run("combine_all_eq30", max_num_iter=30, early_stop=False, only_val_once= False)
    # EstimationValidator.multi_run("combine_k10_le30", max_num_iter=30, early_stop=True)
    # EstimationValidator.multi_run("combine_k10_eq30", max_num_iter=30, early_stop=False)
    # EstimationValidator.multi_run("combine_k50_le30", max_num_iter=30, early_stop=True, k_index= -1)
    # EstimationValidator.multi_run("combine_k50_eq30", max_num_iter=30, early_stop=False, k_index= -1)
    # ClassificationValidator.run(file_name="class_k10_eq30", est_scores_file_name="combine_k10_eq30")
    # ClassificationValidator.run(file_name="class_k50_le30", est_scores_file_name="combine_k50_le30", val_only_once=False)
    # ClassificationValidator.run(file_name="class_mlp_k50_eq30", est_scores_file_name="combine_k50_eq30", run_algos=["fp"])
    # ClassificationValidator.run(file_name="class_mlp_k50_le30", est_scores_file_name="combine_k50_le30")
    # SGDValidator.run_compare()
    EstimationValidator.early_stop()
    # EstimationValidator.multi_run("label_k10_le30", max_num_iter=30, early_stop=True, k_index= -1, dataset_names=['cora', 'computers'] ,run_algos=['umtp', 'umtp_label_25','umtp_label_50', 'umtp_label_75', 'umtp_label_all'])
    # SGDValidator.sgd_params_search()
    # SGDValidator.sgd_params_vector()

    
