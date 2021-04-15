import h5py
import csv
import os


def get_events_and_latency(csv_file_path):
    events_list = []
    try:
        with open(csv_file_path)as f:
            f_csv = csv.reader(f)
            index = 0
            for row in f_csv:
                row_content = row[0].split("\t")
                if index == 0:
                    index = index + 1
                    continue
                elif index != 1 and row_content[2] == "boundary":
                    index = index + 1
                    continue
                elif row_content[2] == "800000" or row_content[2] == "800001":
                    index = index + 1
                    continue
                else:
                    index = index + 1
                    events_dict = {"number": float(row_content[0]), "latency": float(row_content[1]),
                                   "type": row_content[2]}
                    events_list.append(events_dict)
        if events_list[0]["type"] != "boundary":
            events_list.insert(0, {"number": -1, "latency": 0, "type": "boundary"})
        return events_list
    except:
        print("open {} error!".format(csv_file_path))


def flatten(a):
    if not isinstance(a, (list,)):
        return [a]
    else:
        b = []
    for item in a:
        b += flatten(item)
        return b


def split_eeg_by_events(eeg_file_path, save_path, events):
    try:
        eeg_data_mat = h5py.File(eeg_file_path, 'r')
        eeg_file_name, suffix = os.path.splitext(os.path.basename(eeg_file_path))
        eeg_data = eeg_data_mat['data']
        count_11 = 0
        count_22 = 0
        count = 0
        boundary_instead = None
        for item in events[1:]:
            if item["type"] == "11":
                boundary_instead = "22"
                break
            elif item["type"] == "22":
                boundary_instead = "11"
                break
            else:
                continue
        if events[0]["type"] == 'boundary':
            events[0]["type"] = boundary_instead
            events[0]["latency"] = 0
        filename_data = save_path + os.path.sep + eeg_file_name + "_data_split.hdf"
        save_data = h5py.File(filename_data, 'w')
        for i in range(1, len(events) + 1):
            data_split = None
            if i == len(events):
                data_split = eeg_data[int(events[i - 1]["latency"]):]
            else:
                data_split = eeg_data[int(events[i - 1]["latency"]):int(events[i]["latency"])]
            data_split = data_split.reshape(len(data_split), 62)
            data_split = data_split.transpose(1, 0)
            key = None
            if str(events[i - 1]["type"]) == "11":
                count = count + 1
                key = str(events[i - 1]["type"]) + '_' + str(count)
            else:
                count = count + 1
                key = str(events[i - 1]["type"]) + '_' + str(count)
            save_data[key] = data_split
        save_data.close()

    except Exception as e:
        print("split_eeg_by_events error: {}. error message: {}".format(eeg_file_path, e))


def batch_get_split_data(eeg_path, event_path, save_path):
    for home, dirs, eeg_files in os.walk(eeg_path):
        for eeg_file in eeg_files:
            events_file = event_path + os.path.sep + os.path.splitext(eeg_file)[0] + "_events" + ".csv"
            eeg_file_path = eeg_path + os.path.sep + eeg_file
            events = get_events_and_latency(events_file)
            split_eeg_by_events(eeg_file_path, save_path, events)


if __name__ == '__main__':
    # eeg_path = "/Users/weiyong/Desktop/eeg/eeg_mat"
    # event_path = "/Users/weiyong/Desktop/eeg/events"
    # save_path = "/Users/weiyong/Desktop/eeg/save"
    # batch_get_split_data(eeg_path, event_path, save_path)

    # file_path = "/Users/weiyong/Desktop/eeg/events/sub1_095_1_events.csv"
    # events = get_events_and_latency("/Users/weiyong/Desktop/eeg/events/sub1_005_2_events.csv")
    # eeg_file_path = "/Users/weiyong/Desktop/eeg/eeg_mat/sub1_005_2.mat"
    # save_path = "/Users/weiyong/Desktop/eeg/save_2/"
    # split_eeg_by_events(eeg_file_path, save_path, events)
    save_path = "/Users/weiyong/Desktop/eeg/save_2/sub1_005_2_data_split.hdf"
    f = h5py.File(save_path, 'r')
    for i in f.keys():
        print(i)

