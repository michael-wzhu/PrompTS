import copy
import json
import os

task_type_prompt = "Task type: time series classfication"

template_1 = """<query>
在存在噪声数据的情况下，我们需要诊断汽车子系统中是否存在某种症状。每个案例包含500个发动机噪声测量值的时间序列：<ts-data-blank-sep>，其中数据点以空格分隔，请问该案例中，汽车系统是否存在这种症状？
<response>
"""

list_templates_by_chatgpt = [
    """<query>
我们需要在存在噪声数据的情况下诊断汽车子系统是否存在某种症状。每个案例包括一个包含500个发动机噪声测量值的时间序列：<ts-data-blank-sep>，其中数据点以空格分隔。请问在该案例中，汽车系统是否存在这种症状？
<response>
""",
    """<query>
我们需要判断汽车子系统在噪声数据影响下是否出现某种症状。每个案例都有一个包含500个发动机噪音测量值的时间序列：<ts-data-blank-sep>。数据点之间用空格分隔。请问在这个案例中，汽车系统是否存在这种症状？
<response>
""",
    """<query>
In the presence of noise data, we need to diagnose whether a certain symptom exists in the automotive subsystem. Each case consists of a time series of 500 engine noise measurements: <ts-data-blank-sep>, where data points are separated by spaces. Could you please let us know if this symptom exists in the automotive system of this case?
<response>
""",
    """<query>
We need to diagnose the presence of a certain symptom in the automotive subsystem in the presence of noisy data. Each case includes a time series of 500 measurements of engine noise: <ts-data-blank-sep>. The data points are separated by spaces. Could you please inform us if this symptom is present in the automotive system of this particular case?
<response>
""",
    """<query>
We need to decide whether the car subsystem shows a certain symptom under the influence of noisy data. Each case includes a time series of 500 engine noise measurements: <ts-data-blank-sep>, with data points separated by spaces. Could you please inform us if this symptom is present in the automotive system of this particular case?
<response>
""",

]

label_map = {
    "-1": "not exist a certain symptom",
    "1": "exist a certain symptom",
}
label_map_zh = {
    "-1": "不存在某种症状",
    "1": "存在某种症状",
}


list_templates = []
list_templates.append(template_1)
list_templates.extend(list_templates_by_chatgpt)

task_name = "FordB"
to_folder = os.path.join(
    "E:\\硕\\Time_Series_Instruct-main", f"{task_name}"
)
os.makedirs(to_folder, exist_ok=True)

for mode in ["TRAIN", "TEST"]:
    list_datas = []
    with open(f"E:\\硕\\Time_Series_Instruct-main\\UCRArchive_2018\\{task_name}\\{task_name}_{mode}.tsv", "r", encoding="utf-8") as f:
        for row in f:
            row = row.strip()
            if not row:
                continue

            label_ = row.split("\t")[0]
            label_name_ = label_map[label_]
            label_zh_name_ = label_map_zh[label_]

            ts_data_ = row.split("\t")[1: ]
            assert len(ts_data_) == 500
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
