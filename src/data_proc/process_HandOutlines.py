import copy
import json
import os

task_type_prompt = "Task type: time series classfication"


template_1 = """<query>
Could you please help me validate the accuracy of the output labels for the time sequence data: <ts-data-blank-sep>? The data is separated by spaces and represents the hand contour, which has been extracted from images using an algorithm.
<response>
"""

list_templates_by_chatgpt = [
    """<query>
The collection of data in our list signifies a time sequence, comprised of hand contour data extracted by algorithms and delimited by spaces: <ts-data-blank-sep>. Can you determine whether the output labels are correct?
<response>
""",
    """<query>
We have a list of data representing a time sequence, separated by spaces: <ts-data-blank-sep>. It is the hand contour data extracted from images using an algorithm. Can you help me verify if their output labels are correct?
<response>
""",
    """<query>
We possess a list of time sequence data, separated by spaces: <ts-data-blank-sep>, that represents the hand contour extracted from images using an algorithm. Can you assist me in verifying whether the output labels for this data are accurate?
<response>
""",
    """<query>
我们有一个数据列表，表示一个时间序列（用空格分隔）：<ts-data-blank-sep>。它是利用算法提取图像的手部轮廓数据。你能帮助我确定他们的输出标记是否正确吗？
<response>
""",
    """<query>
请你协助确认输出标记是否准确。我们的数据列表表示一个时间序列，它由算法提取出的手部轮廓数据组成（用空格分隔）: <ts-data-blank-sep>。
<response>
""",
    """<query>
我们的列表中收集的数据表示一个时间序列，由算法提取的手部轮廓数据组成，并以空格分隔：<ts-data-blank-sep>。你能确定输出标签是否正确吗？
<response>
""",
]

label_map = {
    "0": "correct",
    "1": "incorrect",
}
label_map_zh = {
    "0": "标记正确",
    "1": "标记不正确",
}


list_templates = []
list_templates.append(template_1)
list_templates.extend(list_templates_by_chatgpt)

task_name = "HandOutlines"
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
            assert len(ts_data_) == 2709
            ts_data_blank_sep = " ".join(ts_data_)
            ts_data_comma_sep = ",".join(ts_data_)

            for template_ in list_templates:
                template_ = copy.copy(template_)

                # 确认数据怎么分割的
                if "comma-sep" in template_:
                    template_.replace(
                        "<ts-data-comma-sep>",
                        ts_data_comma_sep
                    )
                else:
                    template_.replace(
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


    with open(os.path.join(to_folder, f"{mode}.json"), "w", encoding="utf-8") as f:
        for samp in list_datas:
            f.write(
                json.dumps(samp, ensure_ascii=False) + "\n"
            )
