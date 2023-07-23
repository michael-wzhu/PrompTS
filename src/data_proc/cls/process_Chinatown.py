import copy
import json
import os

task_type_prompt = "Task type: time series classfication"

template_1 = """<query>
We have the following time series data (seperated using blank spaces): <ts-data-blank-sep>.
The time series data is the pedestrian count  in Chinatown-Swanston St (North) in each hour of a certain day of the year 2017. Could you determine whether the data is collected on a weekday or a weekend? 
<response>
"""

list_templates_by_chatgpt = [
    """<query>
We have the following time series data (separated using blank spaces): <ts-data-blank-sep>.
The time series data represents the pedestrian count in Chinatown-Swanston St (North) for each hour of a certain day in the year 2017. Can you determine if the data was collected on a weekday or a weekend? 
<response>
""",
    """<query>
Given the time series data provided (separated by blank spaces): <ts-data-blank-sep>, we need to analyze whether it corresponds to a weekday or a weekend. This data represents the pedestrian count in Chinatown-Swanston St (North) for each hour of a specific day in the year 2017. Can you help us determine the day of the week on which this data was collected? 
<response>
""",
    """<query>
We are examining a time series dataset (separated by blank spaces): <ts-data-blank-sep>, which contains the hourly pedestrian count in Chinatown-Swanston St (North) for a particular day in 2017. Our task is to ascertain whether this data was gathered on a weekday or a weekend. Could you assist us in determining the specific day category?
<response>
""",
    """<query>
We have at our disposal a time series dataset (separated using blank spaces): <ts-data-blank-sep>. It captures the pedestrian count in Chinatown-Swanston St (North) every hour for a chosen day in the year 2017. Our objective is to classify this data as either representing a weekday or a weekend. Are you able to help us in identifying the corresponding day category?
<response>
""",
    """<query>
To analyze the data collected on a specific day in 2017, we possess a time series dataset (separated by blank spaces): <ts-data-blank-sep>. This dataset comprises the pedestrian count in Chinatown-Swanston St (North) for each hour. We need to determine if the data corresponds to a weekday or a weekend. Can you provide assistance in classifying the day category based on this data?
<response>
""",
    """<query>
We are investigating a time series dataset (separated using blank spaces): <ts-data-blank-sep>. It contains the pedestrian count in Chinatown-Swanston St (North) for each hour of a specific day in 2017. Our objective is to determine whether the data represents a weekday or a weekend. Could you aid us in categorizing the day based on this information?
<response>
""",
    """<query>
In order to analyze the data collected for a particular day in 2017, we have a time series dataset (separated by blank spaces): <ts-data-blank-sep>. This dataset represents the pedestrian count in Chinatown-Swanston St (North) for each hour. We need to establish whether the data corresponds to a weekday or a weekend. Can you assist us in classifying the day category using this information?
<response>
""",
    """<query>
我们有一个时间序列数据集（用空格分隔）：<ts-data-blank-sep>。它记录了2017年某一天中唐人街斯旺斯顿街（北部）每小时的行人数量。我们需要确定这些数据是在工作日还是周末收集的。你能帮助我们确定具体的日期类别吗？
<response>
""",
    """<query>
我们有一个数据列表，表示一个时间序列（用空格分隔）：<ts-data-blank-sep>。它记录了2017年某一天中唐人街斯旺斯顿街（北部）每小时的行人数量。我们的目标是将这些数据分类为工作日还是周末。你能帮助我们确定具体的日期类别吗？
<response>
""",
    """<query>
We have the following time series data (separated by commas): <ts-data-comma-sep>.
The time series data represents the pedestrian count in Chinatown-Swanston St (North) for each hour of a specific day in the year 2017. Can you help us determine if the data was collected on a weekday or a weekend?
<response>
""",
    """<query>
We possess a time series dataset (comma-separated): <ts-data-comma-sep>. This dataset represents the pedestrian count in Chinatown-Swanston St (North) for each hour of a particular day in the year 2017. Our objective is to determine whether this data was collected on a weekday or a weekend. Can you assist us in classifying the day category?
<response>
""",
    """<query>
We have the following time series data available (comma-separated): <ts-data-comma-sep>. This data captures the pedestrian count in Chinatown-Swanston St (North) for each hour of a specific day in 2017. Our task is to determine if this data was collected on a weekday or a weekend. Can you help us with the classification?
<response>
""",
    """<query>
We possess a time series dataset (comma-separated): <ts-data-comma-sep>. It represents the pedestrian count in Chinatown-Swanston St (North) for each hour of a certain day in 2017. Our goal is to determine whether this data corresponds to a weekday or a weekend. Could you assist us in identifying the day category?
<response>
""",

]

label_map = {
    "1": "weekend",
    "2": "weekday",
}
label_map_zh = {
    "1": "周末",
    "2": "工作日",
}


list_templates = []
list_templates.append(template_1)
list_templates.extend(list_templates_by_chatgpt)

task_name = "Chinatown"
to_folder = os.path.join(
    "datasets/prompt_datasets", f"{task_name}"
)
os.makedirs(to_folder, exist_ok=True)

for mode in ["TRAIN", "TEST"]:
    list_datas = []
    with open(f"datasets/UCRArchive_2018/{task_name}/{task_name}_{mode}.tsv", "r", encoding="utf-8") as f:
        for row in f:
            row = row.strip()
            if not row:
                continue

            label_ = row.split("\t")[0]
            label_name_ = label_map[label_]
            label_zh_name_ = label_map_zh[label_]

            ts_data_ = row.split("\t")[1: ]
            assert len(ts_data_) == 24
            ts_data_blank_sep = " ".join(ts_data_)
            ts_data_comma_sep = ",".join(ts_data_)

            for template_ in list_templates:
                template_ = copy.copy(template_)
                # print(template_)

                # 确认数据怎么分割的
                if "<ts-data-comma-sep>" in template_:
                    template_ = template_.replace(
                        r"<ts-data-comma-sep>",
                        ts_data_comma_sep
                    )
                else:
                    template_ = template_.replace(
                        r"<ts-data-blank-sep>",
                        ts_data_blank_sep
                    )

                assert "<ts-data-comma-sep>" not in template_
                assert "<ts-data-blank-sep>" not in template_

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


    with open(os.path.join(to_folder, f"{mode}.json"), "w", encoding="utf-8") as f:
        for samp in list_datas:
            f.write(
                json.dumps(samp, ensure_ascii=False) + "\n"
            )
