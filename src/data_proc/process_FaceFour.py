import copy
import json
import os

task_type_prompt = "Task type: time series classfication"

template_1 = """<query>
以下时间序列为多个人的脸部轮廓数据：<ts-data-blank-sep>，其中数据点以空格分隔。
数据对应4个人，请判断该数据是属于哪一个人的？
<options>
可能的类别包括：1，2，3，4。
请从上述选项中选择一个回答。
<response>
"""

list_templates_by_chatgpt = [
    """<query>
The following time series represents facial contour data for multiple individuals: <ts-data-blank-sep>, where data points are separated by spaces.
This data corresponds to 4 individuals. Please determine which individual this data belongs to.
<options>
Possible categories include: 1, 2, 3, 4.
Please select one answer from the options provided.
<response>
""",
    """<query>
给定一个时间序列，其中包含多个人的脸部轮廓数据：<ts-data-blank-sep>。每个数据点之间用空格分隔。
这些数据对应于4个人。请确定这些数据属于哪个人。
<options>
以下是可能的选项：1，2，3，4。
请从上述选项中选择一个答案。
<response>
""",
    """<query>
Given a time series that contains facial contour data for multiple individuals: <ts-data-blank-sep>, where data points are separated by spaces.
This data corresponds to 4 individuals. Please determine which individual this data belongs to.
<options>
Here are the possible options: 1, 2, 3, 4.
Please select one answer from the options provided.
<response>
""",
    """<query>
我们有一个时间序列数据集（用空格分隔）：<ts-data-blank-sep>。它表示4个人的脸部轮廓数据，你能帮助我们确定数据是属于哪个人的吗？
<options>
可能的类别包括：1，2，3，4。
请从上述选项中选择一个回答。
<response>
""",
    """<query>
We have a time series dataset (separated by spaces): <ts-data-blank-sep>. It represents facial contour data for 4 individuals. Can you help us determine which person the data belongs to?
<options>
Possible categories include: 1, 2, 3, 4.
Please select an answer from the options provided.
<response>
""",
    """<query>
我们有一个时间序列数据集，其中包含4个人的脸部轮廓数据，数据以空格分隔：<ts-data-blank-sep>。您是否可以帮助我们确定这些数据属于哪个人？
<options>
可能的类别包括：1，2，3，4。
请从上述选项中选择一个答案。
<response>
""",
    """<query>
We have a time series dataset consisting of facial contour data for 4 individuals, with the data separated by spaces: <ts-data-blank-sep>. Can you assist us in determining which person these data belong to?
<options>
Possible categories include: 1, 2, 3, 4.
Please select an answer from the options provided.
<response>
""",


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
list_templates.append(template_1)
list_templates.extend(list_templates_by_chatgpt)

task_name = "FaceFour"
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
            assert len(ts_data_) == 350
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
