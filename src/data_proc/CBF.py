import copy
import json
import os

task_type_prompt = "Task type: time series classfication"


list_templates_by_chatgpt = [
    """<query>
以下时间序列是模拟的数据：<ts-data-blank-sep>，其中数据点以空格分隔。
该时间序列属于某一个具体的类别，其中每个类别是标准正态噪声加上每个类别不同的偏移项，请根据所给的时序数据判断其所属的类别。
<options>
可能的类别包括：1,2,3。
请从上述选项中选择一个回答：
<response>
""",
     """<query>
以下数据是模拟的某一段时间序列：<ts-data-comma-sep>，其中数据点以逗号分隔。
该时间序列属于某一个具体的类别，其中每个类别是标准正态噪声加上每个类别不同的偏移项，请根据所给的时序数据判断其所属的类别。
<options>
可能的类别包括：1,2,3。
请从上述选项中选择一个回答。
<response>
""",
    """<query>
以下时间序列是模拟的数据：<ts-data-blank-sep>，其中数据点以空格分隔。该时间序列属于某一个具体的类别。
其中每个类别是标准正态噪声加上每个类别不同的偏移项。请根判断该时间序列所属的类别。
<options>
可能的类别包括：1,2,3。
请从上述选项中选择一个回答！
<response>
""",
    """<query>
The following time series are simulated data:<ts-data-blank-sep>, where data points are separated by spaces.
This time series belongs to a specific category, where each category is standard normal noise plus different offset terms for each category. Please determine the category it belongs to based on the given time series data.
<options>
Possible categories include: 1,2,3.
Please choose one answer from the above options:
<response>
""",
    """<query>
The following data is a simulated time series:<ts-data-comma-sep>, where data points are separated by commas.
This time series belongs to a specific category, where each category is standard normal noise plus different offset terms for each category. Please determine the category it belongs to based on the given time series data.
<options>
Possible categories include: 1,2,3.
Please choose one answer from the above options.
<response>
""",
    """<query>
The following time series are simulated data:<ts-data-blank-sep>, where data points are separated by spaces. The time series belongs to a specific category.
Each category is standard normal noise plus different offset terms for each category. Please determine the category to which the time series belongs.
<options>
Possible categories include: 1,2,3.
Please choose one answer from the above options!
<response>
"""
]


label_map = {
    "1": "1",
    "2": "2",
    "3": "3",
}

label_map_zh = {
    "1": "1",
    "2": "2",
    "3": "3",
}


list_templates = []
list_templates.extend(list_templates_by_chatgpt)

task_name = "CBF"
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


    