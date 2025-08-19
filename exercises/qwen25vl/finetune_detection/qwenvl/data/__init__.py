import re

OBJECT365 = {
    "annotation_path": "/purestorage/AILAB/AI_1/tyk/3_CUProjects/language_model/VLM/qwen25vl/finetune_detection/messages_output.json",
    "data_path": "/purestorage/AILAB/AI_1/dataset/Objects365/images/train",
}

data_dict = {
    "object365": OBJECT365
}

def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names): # list of dict 형태로 {'annotation_path': '/purestorage/AILAB/AI_1/tyk/3_CUProjects/VLM/messages_output.json', 'data_path': '/purestorage/AILAB/AI_1/dataset/Objects365/images/train', 'sampling_rate': 1.0}
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["object365"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)