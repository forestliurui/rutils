import json

def dict_to_json_file(dict_obj, file_name):
    json.dump(dict_obj, open(file_name, 'w'), indent=2)

def json_file_to_dict(file_name):
    dict_obj = json.load(open(file_name, 'r'))
    return dict_obj

def parser_args_to_dict(args):
    return vars(args)