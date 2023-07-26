import copy
import datetime
import json
import os
import random

from tqdm import tqdm

task_type_prompt = "Task type: time series classfication"

# [DATE-1]： 包含日期，也可以包含星期几
template_1 = {
    "input": """<query>
Between [TIME-POINT-1] and [TIME-POINT-2], the mean daily temperatures recorded in [REGION] were <ts-data-blank-sep> (F) with -99 indicating missing data points. What is the projected temperature on [TIME-POINT-3]?
<response>
""",
    "target": """The anticipated temperature will be [TARGET] degrees."""
}

list_templates_by_chatgpt = [
    {
        "input": """<query>
Between [TIME-POINT-1] and [TIME-POINT-2], the mean daily temperatures recorded in [REGION] amounted to <ts-data-blank-sep> (degrees). What is the projected temperature for [TIME-POINT-3]?
<response>
""",
        "target": "The anticipated temperature will be [TARGET] degrees.",
    },
    {
        "input": """<query>
For the time span covering [TIME-POINT-1] to [TIME-POINT-2], [REGION] experienced an average daily temperature of <ts-data-comma-sep> (F) with -99 denoting unavailable data points. What is the forecasted temperature on [TIME-POINT-3]?
<response>
        """,
        "target": "The forecasted temperature on [TIME-POINT-3] will be [TARGET] degrees.",
    },
{
        "input": """<query>
Considering the period from [TIME-POINT-1] to [TIME-POINT-2], the average daily temperatures observed in [REGION] amounted to <ts-data-blank-sep> (F) with -99 representing missing data. What is the expected temperature on [TIME-POINT-3]?
<response>
        """,
        "target": "The expected temperature will be [TARGET] degrees.",
    },
{
        "input": """<query>
从[TIME-POINT-1]到[TIME-POINT-2]，[REGION]地区的平均每日温度为<ts-data-blank-sep>（华氏度）。-99表示缺失值。请问在[TIME-POINT-3]的温度将是多少？
<response>
        """,
        "target": "温度将会是[TARGET]度。",
    },
{
        "input": """<query>
[TIME-POINT-1]至[TIME-POINT-2]期间，[REGION]地区的平均每日温度为<ts-data-blank-sep>（华氏度）。-99代表有缺失值。请问在[TIME-POINT-3]时的温度会是多少？
<response>
        """,
        "target": "温度会是[TARGET]度。",
    },

]


list_templates = []
list_templates.append(template_1)
list_templates.extend(list_templates_by_chatgpt)

task_name = "udayton_temperature"
to_folder = os.path.join(
    "datasets/prompt_datasets", f"{task_name}"
)
os.makedirs(to_folder, exist_ok=True)

location_id2name = json.load(
    open("datasets/udayton_temperature_archive/location_name.json", "r", encoding="utf-8")
)

window_sizes = [15, 30, 60]
horizon_sizes = [1, 2, 3, 7]

month_id2name = {
    "1": "January",
    "2": "February",
    "3": "March",
    "4": "April",
    "5": "May",
    "6": "June",
    "7": "July",
    "8": "August",
    "9": "September",
    "10": "October",
    "11": "November",
    "12": "December",
}

weekday_idx2name = {
    "1": "Monday",
    "2": "Tuesday",
    "3": "Wednesday",
    "4": "Thursday",
    "5": "Friday",
    "6": "Saturday",
    "7": "Sunday",
}


for location_idx, location_name in tqdm(location_id2name.items()):
    location_name_str = ", ".join(location_name)

    list_time_points = []
    with open(f"datasets/udayton_temperature_archive/allsites/{location_idx}.txt", "r", encoding="utf-8") as f:
        for row in f:
            row = row.strip()
            if not row:
                continue

            # print(row.split(" "))
            row = row.split(" ")
            row = [w.strip() for w in row if len(w.strip()) > 0]
            # print(row)

            list_time_points.append(row)


    for win_size in window_sizes:
        for horizon_size in horizon_sizes:
            if win_size < 40 and horizon_size > 3:
                continue

            list_samples = []
            for start_point in tqdm(range(len(list_time_points) - win_size - horizon_size)):
                end_point = start_point + win_size - 1
                horizon_point = start_point + win_size - 1 + horizon_size

                # 表达时间： May 27, 2019, Monday； May 27, 2019； 2019-05-27
                start_time_info = list_time_points[start_point][: 3]
                end_time_info = list_time_points[end_point][: 3]
                horizon_time_info = list_time_points[horizon_point][: 3]

                list_time_strs = []
                for time_info in [start_time_info, end_time_info, horizon_time_info]:
                    s_month, s_day, s_year = time_info
                    start_time_point_1 = f"{month_id2name[s_month]} {s_day}, {s_year}"
                    start_time_point_2 = f"{month_id2name[s_month][: 3]} {s_day}, {s_year}"

                    s_month = "0" + s_month if len(s_month) == 1 else s_month
                    s_day = "0" + s_day if len(s_day) == 1 else s_day
                    s_date = datetime.date.fromisoformat(f"{s_year}-{s_month}-{s_day}")
                    start_time_point_3 = s_date.strftime("%Y-%m-%d")

                    weekday_idx = s_date.isoweekday()

                    weekday = weekday_idx2name[str(weekday_idx)]
                    # print(weekday)
                    start_time_point_4 = f"{weekday}, " + start_time_point_1
                    start_time_point_5 = f"{weekday}, " + start_time_point_2
                    start_time_point_6 = f"{weekday}, " + start_time_point_3

                    list_time_strs.append(
                        [
                            start_time_point_1, start_time_point_2, start_time_point_3,
                            start_time_point_4, start_time_point_5, start_time_point_6,
                        ]
                    )

                tmp_idx = random.choice(list(range(6)))
                start_time_str = list_time_strs[0][tmp_idx]
                end_time_str = list_time_strs[1][tmp_idx]
                horizon_time_str = list_time_strs[2][tmp_idx]
                # print("start_time_str: ", start_time_str)
                # print("end_time_str: ", end_time_str)
                # print("horizon_time_str: ", horizon_time_str)

                ts_data_ = [w[-1] for w in list_time_points[start_point: end_point + 1]]
                horizon_data_ = list_time_points[horizon_point][-1]
                ts_data_blank_sep = " ".join(ts_data_)
                ts_data_comma_sep = ",".join(ts_data_)


                for template_ in list_templates:
                    template_ = copy.copy(template_)
                    # print(template_)
                    query_template = template_["input"]
                    response_template = template_["target"]

                    '''
                    From [TIME-POINT-1] to [TIME-POINT-2], the average daily temperatures of [REGION] was <ts-data-blank-sep> (degree). What is the temperature going to be on [TIME-POINT-3]?
                    '''

                    # 确认数据怎么分割的
                    if "<ts-data-comma-sep>" in query_template:
                        query_template = query_template.replace(
                            r"<ts-data-comma-sep>",
                            ts_data_comma_sep
                        )
                    else:
                        query_template = query_template.replace(
                            r"<ts-data-blank-sep>",
                            ts_data_blank_sep
                        )
                    query_template = query_template.replace(
                        "[TIME-POINT-1]", start_time_str
                    ).replace(
                        "[TIME-POINT-2]", end_time_str
                    ).replace(
                        "[TIME-POINT-3]", horizon_time_str
                    ).replace(
                        "[REGION]", location_name_str
                    )

                    assert "<ts-data-comma-sep>" not in query_template
                    assert "<ts-data-blank-sep>" not in query_template

                    # 确认中文还是英文
                    response_template = response_template.replace(
                        "[TARGET]", horizon_data_
                    ).replace(
                        "[TIME-POINT-3]", horizon_time_str
                    )

                    list_samples.append(
                        {
                            "original_data": (ts_data_, horizon_data_),
                            "target": response_template,
                            "input": query_template,
                            "task_name": task_name,
                            "task_type": "forecast",
                        }
                    )


            with open(os.path.join(to_folder, f"{location_idx}_{win_size}_{horizon_size}.json"), "w", encoding="utf-8") as f:
                for samp in list_samples:
                    f.write(
                        json.dumps(samp, ensure_ascii=False) + "\n"
                    )
