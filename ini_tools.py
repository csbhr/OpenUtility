import configparser


def get_ini_value(filepath, section_name, key_name):
    conf = configparser.ConfigParser()
    conf.read(filepath)
    return conf.get(section_name, key_name)
