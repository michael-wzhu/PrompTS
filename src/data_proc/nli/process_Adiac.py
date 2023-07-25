import copy
import json
import os
import random

task_type_prompt = "Task type: time series inference"

template_1 = """<query>
In order to automatically identify diatoms (unicellular algae), the outlines are extracted from thresholded images, and the following two time series are generated as distances to a reference point: <ts-data-blank-1>, and <ts-data-blank-2>. 
Could you determine whether the two time series represent the same type of diatoms or not? 
<options>
The answering options are: yes, no.
<response>
"""

list_templates_by_chatgpt = [
    """<query>
To automate the identification process of diatoms (unicellular algae), the system extracts outlines from thresholded images, resulting in the creation of two time series: <ts-data-blank-1>, and <ts-data-blank-2>. These time series represent distances to a reference point.
Your task is to determine whether both time series correspond to the same type of diatoms or not.
<options>
Please select one of the following answering options: "yes" or "no".
<response>
    """,
    """<query>
Our objective is to automate the recognition of diatoms (unicellular algae) by extracting their outlines from thresholded images. The resulting data yields two time series: <ts-data-blank-1> and <ts-data-blank-2>, both representing distances to a reference point. Your task is to ascertain whether these two time series pertain to the same type of diatoms or not.
<options>
Kindly provide your response by selecting either "yes" or "no".
<response>
    """,
    """<query>
In order to facilitate the automated identification of diatoms (unicellular algae), we extract outlines from thresholded images and create two time series: <ts-data-blank-1> and <ts-data-blank-2>. These time series represent the distances to a reference point. Your task is to determine whether both time series correspond to the same type of diatoms or not.
<options>
Please select either "yes" or "no" as your answer.
<response>
    """,
    """<query>
We have implemented an automated diatom (unicellular algae) identification system that derives two time series: <ts-data-blank-1> and <ts-data-blank-2>. These time series are based on distances to a reference point, extracted from thresholded images. Your role is to assess whether the two time series belong to the same type of diatoms or not.
<options>
Kindly respond with either "yes" or "no".
<response>
    """,
    """<query>
As part of our diatom (unicellular algae) recognition automation process, we extract outlines from thresholded images, generating two time series: <ts-data-blank-1> and <ts-data-blank-2>. These time series represent distances to a reference point. Your task is to determine whether both time series correspond to the same type of diatoms or not.
<options>
Please indicate your response with either "yes" or "no".
<response>
    """,
    """<query>
To achieve automated diatom (unicellular algae) identification, we extract outlines from thresholded images and obtain two time series: <ts-data-blank-1> and <ts-data-blank-2>. These time series represent distances to a reference point. Your role is to determine whether both time series belong to the same type of diatoms or not.
<options>
Kindly select "yes" or "no" as your response.
<response>
    """,
    """<query>
为了自动识别硅藻（单细胞藻类），系统从阈值处理后的图像中提取轮廓，得到以下两个时间序列作为到参考点的距离：<ts-data-blank-1> 和 <ts-data-blank-2>。请您判断这两个时间序列是否代表同一类型的硅藻？
<options>
请在以下选项中选择：是、否。
<response>
    """,
    """<query>
我们的目标是通过从阈值处理后的图像中提取硅藻（单细胞藻类）的轮廓来实现自动识别。这个过程生成了两个时间序列：<ts-data-blank-1> 和 <ts-data-blank-2>，它们分别表示到参考点的距离。您的任务是判断这两个时间序列是否对应同一类型的硅藻？
<options>
请您选择以下答案：是、否。
<response>
    """,
    """<query>
为了实现硅藻（单细胞藻类）的自动识别，我们从阈值处理后的图像中提取了轮廓，并得到两个时间序列：<ts-data-blank-1> 和 <ts-data-blank-2>，它们代表了到参考点的距离。请您判断这两个时间序列是否属于同一类型的硅藻？
<options>
请用 "是" 或 "否" 来回答。
<response>
    """,

]

label_map = {
    "1": "yes",
    "0": "no",
}
label_map_zh = {
    "1": "是",
    "0": "不是",
}


list_templates = []
list_templates.append(template_1)
list_templates.extend(list_templates_by_chatgpt)

task_name = "Adiac"
to_folder = os.path.join(
    "datasets/prompt_datasets", f"{task_name}"
)
os.makedirs(to_folder, exist_ok=True)

for mode in ["TRAIN", "TEST"]:

    # load original data
    dict_orig_label2samples = {}
    with open(f"datasets/UCRArchive_2018/{task_name}/{task_name}_{mode}.tsv", "r", encoding="utf-8") as f:
        for row in f:
            row = row.strip()
            if not row:
                continue

            label_ = row.split("\t")[0]
            ts_data_ = row.split("\t")[1:]
            ts_data_blank_sep = "[" + " ".join(ts_data_) + "]"
            if label_ not in dict_orig_label2samples:
                dict_orig_label2samples[label_] = []
            dict_orig_label2samples[label_].append(ts_data_blank_sep)

    list_datas = []
    for orig_label in dict_orig_label2samples.keys():

        for _ in range(len(dict_orig_label2samples[orig_label]) * 4):

            label_a = orig_label
            label_b = random.choice(list(set(dict_orig_label2samples.keys()).difference(set(label_a))))
            # print(label_a, label_b)

            sample_a_1, sample_a_2 = random.sample(dict_orig_label2samples[label_a], 2)
            sample_b_1 = random.choice(dict_orig_label2samples[label_b])
            # print(sample_a_1)

            # 挑选一个模板
            template_ = random.choice(list_templates)
            # print(template_)

            sample_1 = copy.copy(template_)
            target_1 = ""
            if "序列" in template_:
                target_1 = label_map_zh["1"]
            else:
                target_1 = label_map["1"]

            # 确认数据怎么分割的
            if "data-comma" in sample_1:
                sample_1 = sample_1.replace(
                    "<ts-data-comma-1>",
                    sample_a_1
                ).replace(
                    "<ts-data-comma-2>",
                    sample_a_2
                )
            else:
                assert "data-blank" in sample_1
                sample_1 = sample_1.replace(
                    "<ts-data-blank-1>",
                    sample_a_1
                ).replace(
                    "<ts-data-blank-2>",
                    sample_a_2
                )

            list_datas.append(
                {
                    "original_data": (sample_a_1, label_a, sample_a_2, label_a),
                    "target": target_1,
                    "input": sample_1,
                    "task_name": task_name,
                    "task_type": "nli",
                }
            )

            sample_2 = copy.copy(template_)
            target_2 = ""
            if "序列" in template_:
                target_2 = label_map_zh["0"]
            else:
                target_2 = label_map["0"]

            # 确认数据怎么分割的
            if "data-comma" in sample_2:
                sample_2 = sample_2.replace(
                    "<ts-data-comma-1>",
                    sample_a_1
                ).replace(
                    "<ts-data-comma-2>",
                    sample_b_1
                )
            else:
                sample_2 = sample_2.replace(
                    "<ts-data-blank-1>",
                    sample_a_1
                ).replace(
                    "<ts-data-blank-2>",
                    sample_b_1
                )

            list_datas.append(
                {
                    "original_data": (sample_a_1, label_a, sample_b_1, label_b),
                    "target": target_2,
                    "input": sample_2,
                    "task_name": task_name,
                    "task_type": "nli",
                }
            )

    with open(os.path.join(to_folder, f"{mode}.json"), "w", encoding="utf-8") as f:
        for samp in list_datas:
            f.write(
                json.dumps(samp, ensure_ascii=False) + "\n"
            )
