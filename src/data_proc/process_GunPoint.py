import copy
import json
import os

task_type_prompt = "Task type: time series classfication"

template_1 = """<query>
The following time series represents the centroid of the actor's right hands on the x-axis while they are making a motion with their hand: <ts-data-blank-sep>. The data points are separated by spaces. 
Can you determine whether the action depicted is drawing a gun or simply pointing a finger at a target?
<response>
"""

list_templates_by_chatgpt = [
    """<query>
The given time series illustrates the midpoint of the actor's right hand movements along the x-axis during a gesture: <ts-data-blank-sep>. The data points are spaced out.
Can you ascertain if the portrayed action involves drawing a firearm or merely pointing a finger towards a target?
<response>
""",
    """<query>
我们有一个数据列表，表示一个时间序列（用空格分隔）：<ts-data-blank-sep>。它记录了一个人在用手做动作时右手在x轴上的质心。我们的目标是根据这些数据判断这个人的动作是在拔枪还是仅是用手指向某个目标。
你能帮助我们判断一下吗？
<response>
""",
    """<query>
下面的时间序列表示演员用手做动作时右手在x轴上的质心移动：<ts-data-blank-sep>。数据点之间用空格分隔。
你能确定画面描绘的动作是在掏枪还是只是用手指着目标吗?
<response>
""",
    """<query>
Can you help us classify the given time series dataset (separated by spaces): <ts-data-blank-sep>? It contains the movement data of a person's right hand centroid. Our goal is to determine what actions the data represents, with two categories: drawing a gun and pointing.
<response>
""",
    """<query>
Could you assist us in categorizing the provided time series dataset (separated by spaces): <ts-data-blank-sep>? It comprises the motion information of the centroid of an individual's right hand. Our objective is to identify the actions presented in the data, classifying them into two categories: drawing a firearm and pointing.
<response>
""",
    """<query>
May we seek your assistance in categorizing the given time series dataset (separated by spaces): <ts-data-blank-sep>? It encompasses the movement information of a person's right hand centroid. We aim to identify the actions represented by the data, classifying them into two categories: drawing a gun and pointing.
<response>
""",
    """<query>
We have a data list representing a time series (separated by spaces): <ts-data-blank-sep>. It records the centroid of a person's right hand on the x-axis while performing an action with their hand. Our objective is to determine whether the person's action is drawing a gun or simply pointing towards a target based on this data. Can you assist us in making this judgment?
<response>
""",
    """<query>
We possess a data list that signifies a chronological sequence (separated by spaces): <ts-data-blank-sep>. It registers the centroid of an individual's right hand on the x-axis during manual movements. Our aim is to assess whether these movements indicate drawing a firearm or merely directing towards a specific target. Could you aid us in making this evaluation?
<response>
""",
]

label_map = {
    "1": "gun",
    "2": "no gun (pointing)",
}
label_map_zh = {
    "1": "拔枪",
    "2": "无枪（用手指指着）",
}


list_templates = []
list_templates.append(template_1)
list_templates.extend(list_templates_by_chatgpt)

task_name = "GunPoint"
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
            assert len(ts_data_) == 150
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
