from os.path import exists

class Writer:
    def __init__(self, file_path, delete=True):
        self.file_path = file_path
        if file_path is not None:
            if not exists(self.file_path) or delete:
                f = open(file_path, 'w')
                f.write('')
                f.close()

    def append(self, text):
        text = str(text)
        with open(self.file_path, 'a') as f:
            f.write(text + '\t')
            f.close()

    def append_line(self, line):
        if self.file_path is None:
            print(str(line))
        else:
            line = str(line)+'\n'
            with open(self.file_path, 'a') as f:
                f.write(line)
                f.close()


def str_to_int(string):
    try:
        return int(string)
    except ValueError:
        return None


def str_to_float(string):
    try:
        return float(string)
    except ValueError:
        return None


def split_list_str(string, split_str, last_type=float):
    string = string.split(split_str)
    if split_str == ',':
        return [last_type(el) for el in string]
    else:
        split_str = split_str[1:-1]
        return [split_list_str(el, split_str, last_type) for el in string]


def str_to_float_list(string):
    try:
        brace_count = 0
        for brace_count in range(len(string)):
            if string[brace_count] != '[':
                break
        if brace_count == 0:
            return None
        string_removed_braces = string[brace_count:-brace_count]
        split_str = ']'*(brace_count-1) + ',' + '['*(brace_count-1)
        return split_list_str(string_removed_braces, split_str, float)
    except:
        return None


def fix_lists_in_line(line):
    found_starting_brace = 0
    starting_index = -1
    last_ending_index = 0
    result = ''
    for idx in range(len(line)):
        if line[idx] == '[':
            if starting_index == -1:
                if line[idx - 7:idx] == 'tensor(':
                    result += line[last_ending_index:idx-7]
                else:
                    result += line[last_ending_index:idx]
                starting_index = idx
            found_starting_brace += 1
        elif line[idx] == ']':
            found_starting_brace -= 1
            if found_starting_brace == 0:
                result += line[starting_index:idx].replace(' ', '')
                starting_index = -1
                last_ending_index = idx
    if starting_index != -1:
        result += line[starting_index:].replace(' ', '')
    else:
        if last_ending_index != -1:
            result += line[last_ending_index:]
    return result

def log_line_to_dict(line):
    line = fix_lists_in_line(line)
    line = line.replace('\r', '').replace('\n', '')
    line = line.split(', ')
    result = dict()
    for entry in line:
        entry = entry.split('=')
        if len(entry) == 2:
            param = entry[0]
            value = entry[1]
            # convert value to int, float or list of floats
            converted_value = str_to_int(value)
            if converted_value is None:
                converted_value = str_to_float(value)
                if converted_value is None:
                    converted_value = str_to_float_list(value)
                    if converted_value is None:
                        converted_value = value
            result[param] = converted_value
    return result


def check_dict_for_attributes(dictionary, attributes):
    if len(dictionary.keys()) == 0:
        return False
    for attribute in attributes.keys():
        if attribute not in dictionary.keys():
            return False
    for param, value in attributes.items():
        if dictionary[param] != value and value != '*':
            return False
    return True


def read_log(file_path, attributes):    # schmidt_rank, num_points, std=0, U=None, dataset=None):
    result = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            line_dict = log_line_to_dict(line)
            # add line_dict only if all the attributes are contained
            if check_dict_for_attributes(line_dict, attributes):
                result.append(line_dict)
    # print(f"length of subresult = {len(result)}")
    return result


def read_logs_regex(file_reg, attributes):
    import glob
    result = []
    for file_path in glob.glob(file_reg):
        result.extend(read_log(file_path, attributes))
    return result


if __name__ == '__main__':
    r = \
    [
        [
            [1, 2],
            [3, 4]
        ],
        [
            [5, 6],
            [7, 8]
        ]
    ]
    r = str(r).replace(' ', '')
    print(str_to_float_list(r))
