import json
import re
import glob
import itertools

# Util functions for visualisation and analysis of experiment results

# One line in the results files usually contains:
# dict_keys(['schmidt_rank', 'num_points', 'std', 'losses', 'risk', 'train_time', 'qnn', 'unitary'])
def parse_result_line(resline):
    """Parses one line of an experiment results file"""
    m = re.sub(r"([a-zA-Z_]{2,})", "\"\\1\"", resline) # match at least 2 characters to prevent matchin "e" in scientific notation
    m = re.sub(r"=", r":", m)
    jsonline = "{" + m + "}"
    try:
        return json.loads(jsonline)
    except json.decoder.JSONDecodeError as e:
        print("Error line: ")
        print(jsonline)
        print(e)


def parse_process_file(filename):
    """Parses one process files in the experiment results"""
    file = open(filename, 'r')
    count = 0

    expdata_lines = []

    while True:
        line = file.readline()
        if not line:
            break
        
        if line.startswith('schmidt_rank'): # proper result line (others might contain different info)
            expdata_lines.append(line)

    file.close()    

    return [parse_result_line(line) for line in expdata_lines]

def parse_process_directory(dirname):
    """Parses process directory in experiment results - contains results of multiple processes of the same experiment type"""
    resultfiles = glob.glob(dirname + '/result_*.txt')
    
    return list(itertools.chain.from_iterable([parse_process_file(file) for file in resultfiles])) # merge the individual result lists

def get_risks(results):
    """Extracts risks from results"""
    return [res["risk"] for res in results]

def get_dev_from_average(results, average):
    """Computes deviation from average risk for each experiment result"""
    risks = np.array(get_risks(results))
    dev = np.abs(risks - average)
    return dev

def get_max_dev_from_average(results, average):
    """Computes maximal deviation from average risk for experiment results"""
    return np.max(get_dev_from_average(results, average))

def average_risk(results):
    """Computes average risk of all given results"""
    return sum(get_risks(results))/float(len(results))

def get_final_losses(results):
    """Computes list of losses after training for results"""
    return [res["losses"][-1] for res in results]  # final loss is last element in losses

def average_final_loss(results):
    """Computes average loss after training from results"""
    return sum(get_final_losses(results))/float(len(results))
    
def get_dev_from_average(results, average):
    risks = np.array(get_risks(results))
    dev = np.abs(risks - average)
    return dev

def get_max_dev_from_average(results, average):
    return np.max(get_dev_from_average(results, average))

def get_std_dev_risks(results):
    return np.std(get_risks(results))

def get_std_dev_losses(results):
    return np.std(get_final_losses(results))