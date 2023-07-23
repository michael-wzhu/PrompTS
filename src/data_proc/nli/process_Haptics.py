import copy
import json
import os
import random

task_type_prompt = "Task type: time series classfication"


template_1 = """<query>
Time series data of the x-axis movement are recorded when a person enters his/her passgraph (a code to access a system protected by a graphical authentication system) on a touchscreen.
Now given the following two time series:
Time series 1: <ts-data-blank-1>
Time series 2: <ts-data-blank-2>
Could you tell whether the two series are from the same person or not?
<explanation>
Assuming different persons have different passgraphs.
<options>
The answering options are: yes, no.
<response>
"""

list_templates_by_chatgpt = [
    """<query>
We have collected time series data that records the x-axis movement when a person enters their passgraph (a code to access a system protected by a graphical authentication system) on a touchscreen.
Now, we are provided with the following two time series:
Time series 1: <ts-data-blank-1>
Time series 2: <ts-data-blank-2>
Our task is to determine whether these two series originate from the same person or not.
<explanation>
It is assumed that distinct individuals possess unique passgraphs.
<options>
You can respond with either "yes" or "no."
<response>
""",
    """<query>
In this analysis, we have recorded time series data of the x-axis movement when a person enters their passgraph on a touchscreen. We are presented with two time series for evaluation:
Time series 1: <ts-data-blank-1>
Time series 2: <ts-data-blank-2>
Our objective is to determine whether these two series belong to the same individual or not.
<explanation>
It is presumed that distinct individuals possess unique passgraphs.
<options>
The available response options are: yes, no.
<response>
""",
    """
    <query>
We have gathered time series data that captures the x-axis movement when a person enters their passgraph on a touchscreen. The task at hand involves comparing the following two time series:
Time series 1: <ts-data-blank-1>
Time series 2: <ts-data-blank-2>
Our objective is to ascertain whether these two series originate from the same individual.
<explanation>
It is presumed that each person possesses a distinct passgraph, setting them apart from others.
<options>
The available response choices are: yes, no.
<response>
""",
    """<query>
Our data consists of time series detailing the x-axis movement during the entry of a passgraph on a touchscreen. We are now presented with two specific time series for analysis:
Time series 1: <ts-data-blank-1>
Time series 2: <ts-data-blank-2>
The primary goal is to determine if both these series belong to the same individual.
<explanation>
It is assumed that each person has a unique passgraph, allowing differentiation between individuals.
<options>
You can respond with either: yes, no.
<response>
""",
    """<query>
我们记录了一个人在触摸屏上输入他/她的轨迹时，x轴移动的时间序列数据。现在给定以下两个时间序列：
时间序列1：<ts-data-blank-1>
时间序列2：<ts-data-blank-2>
请问这两个序列是否来自同一个人？
<explanation>
假设不同的人有不同的轨迹。
<options>
回答选项为：是，否。
<response>
""",
    """<query>
我们记录了一个人在触摸屏上输入他/她的轨迹时，x轴移动的时间序列数据。现在给定以下两个时间序列：
时间序列1：<ts-data-blank-1>
时间序列2：<ts-data-blank-2>
请问这两个序列是否属于同一个人？
<explanation>
假设不同的个体具有不同的轨迹。
<options>
可供选择的回答是：是，否。
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


task_name = "Haptics"
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
