import copy
import json
import os

task_type_prompt = "Task type: time series classfication"


list_templates_by_chatgpt = [
    """<query>
以下时间序列是取自多个躯干表面部位的心电图数据，对应4个不同的人。
时间序列：<ts-data-blank-sep>，其中数据点以空格分隔。
请判断其是属于哪一个人的心电图。
<options>
可能的类别包括：1,2,3,4。
请从上述选项中选择一个回答：
<response>
""",
     """<query>
以下时间序列是取自多个躯干表面部位的心电图数据，对应4个不同的人。
时间序列：<ts-data-comma-sep>，其中数据点以逗号分隔。
请判断该时间序列是属于哪一个人的心电图数据。
<options>
可能的类别包括：1,2,3,4。
请从上述选项中选择一个回答：
<response>
""",
    """<query>
时间序列：<ts-data-blank-sep>，其中数据点以空格分隔。
上面的时间序列是取自多个躯干表面部位的心电图数据，对应4个不同的人。
请判断其是属于哪一个人的心电图。
<options>
可能的类别包括：1,2,3,4。
请从上述选项中选择一个回答：
<response>
""",
    """<query>
The following time series are ECG data taken from multiple torso surface sites, corresponding to 4 different individuals.
Time series: <ts-data-blank-sep>, where data points are separated by spaces.
Please determine which person's ECG it belongs to.
<options>
Possible classes include: 1,2,3,4.
Please select an answer from the options above:
<response>
""",
    """<query>
The following time series are ECG data taken from multiple torso surface sites, corresponding to 4 different individuals.
Time series: <ts-data-comma-sep>, where data points are separated by commas.
Please determine which person's ECG data this time series belongs to.
<options>
Possible classes include: 1,2,3,4.
Please select an answer from the options above:
<response>
""",
    """<query>
Time series: <ts-data-blank-sep>, where data points are separated by spaces.
The time series above are ECG data taken from multiple torso surface sites, corresponding to 4 different individuals.
Please determine which person's ECG it belongs to.
<options>
Possible classes include: 1,2,3,4.
Please select an answer from the options above:
<response>
"""
]

label_map = {
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
}

label_map_zh = {
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
}


list_templates = []
list_templates.extend(list_templates_by_chatgpt)

task_name = "CinCECGTorso"
to_folder = os.path.join(
    "UCRArchive_2018/prompt_datasets/", f"{task_name}"
)
os.makedirs(to_folder, exist_ok=True)

for mode in ["TRAIN", "TEST"]:
    list_datas = []
    with open(f"UCRArchive_2018/{task_name}/{task_name}_{mode}.tsv", "r", encoding="utf-8") as f:
        for row in f:
            row = row.strip()
            if not row:
                continue

            label_ = row.split("\t")[0]
            label_name_ = label_map[label_]
            label_zh_name_ = label_map_zh[label_]

            ts_data_ = row.split("\t")[1: ]
            # assert len(ts_data_) == 96
            ts_data_blank_sep = " ".join(ts_data_)
            ts_data_comma_sep = ",".join(ts_data_)

            for template_ in list_templates:
                template_ = copy.copy(template_)

                # 确认数据怎么分割的
                if "comma-sep" in template_:
                    template_ = template_.replace(
                        "<ts-data-comma-sep>",
                        ts_data_comma_sep
                    )
                else:
                    template_ = template_.replace(
                        "<ts-data-blank-sep>",
                        ts_data_blank_sep
                    )

                # 确认中文还是英文
                target = ""
                if "序列" in template_:
                    target = label_zh_name_
                else:
                    target = label_name_

                list_datas.append(
                    {
                        "original_data": (label_name_, ts_data_blank_sep),
                        "target": target,
                        "input": template_,
                        "task_name": task_name,
                        "task_type": "classification",
                    }
                )

    print(f"len(list_datas): {len(list_datas)}")

    with open(os.path.join(to_folder, f"{mode}.json"), "w", encoding="utf-8") as f:
        for samp in list_datas:
            f.write(
                json.dumps(samp, ensure_ascii=False) + "\n"
            )


    