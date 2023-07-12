import copy
import json
import os

task_type_prompt = "Task type: time series classfication"

template_1 = """<query>
以下是意大利12个月的电力需求时间序列：<ts-data-blank-sep>，其中数据点以空格分隔。请确定这一天是十月至三月的还是四月至九月的。
<option>
只能从十月至三月、四月至九月中挑一个回答。
不可以回答单独一个月份。
<response>
"""

list_templates_by_chatgpt = [
    """<query>
以下是意大利12个月的电力需求时间序列：<ts-data-blank-sep>，其中数据点以空格分隔。请确定这一天是属于十月至三月还是四月至九月的时间段。
<option>
只能从十月至三月、四月至九月中选择一个答案。
不能回答单独的月份。
<response>
""",
    """<query>
给出意大利的12个月电力需求时间序列：<ts-data-blank-sep>，数据点之间以空格分隔。请确定该日期是在十月至三月之间还是四月至九月之间。
<option>
只能选择十月至三月或四月至九月其中之一。
不能单独回答一个月份。
<response>
""",
    """<query>
Below is the time series of electricity demand for 12 months in Italy: <ts-data-blank-sep>, where data points are separated by spaces. Please determine if this day belongs to October to March or April to September.
<option>
You can only choose between October to March or April to September.
You cannot answer with an individual month.
<response>
""",
    """<query>
Here is the time series of electricity demand for 12 months in Italy: <ts-data-blank-sep>, where data points are separated by spaces. Please determine whether this day falls within the period of October to March or April to September.
<option>
You can only choose between October to March or April to September.
Do not provide an answer with an individual month.
<response>
""",
    """<query>
Provided is the time series of electricity demand for 12 months in Italy: <ts-data-blank-sep>, with data points separated by spaces. Please determine whether this day falls within the period from October to March or from April to September.
<option>
You can only choose between the period from October to March or from April to September.
Do not provide an answer with an individual month.
<response>
""",

]

label_map = {
    "1": "from Oct to March",
    "2": "from April to September",
}
label_map_zh = {
    "1": "十月至三月",
    "2": "四月至九月",
}


list_templates = []
list_templates.append(template_1)
list_templates.extend(list_templates_by_chatgpt)

task_name = "ItalyPowerDemand"
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
