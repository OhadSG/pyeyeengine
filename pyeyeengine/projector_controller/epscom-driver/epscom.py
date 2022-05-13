import subprocess

EPSCOM_CMD = 'epscom-cmd'

def full_command(cmd):
    return "./%s \"%s\"" %(EPSCOM_CMD, cmd)

def epscom_set(cmd):
    subprocess.call(full_command(cmd), shell=True)

def epscom_get(cmd):
    answer = subprocess.check_output(full_command(cmd), shell=True)
    return clean_returned_byte_string(answer)

def clean_returned_byte_string(answer):
    return answer.decode("utf-8").replace("\n", "").split("=")


if __name__ == '__main__':
    answer = epscom_get("LAMP?")
    assert(answer[0]=="LAMP")
    print(answer)

    answer = epscom_get("PWR?")
    assert(answer[0]=="PWR")
    print(answer)

    answer = epscom_get("SNO?")
    assert (answer[0] == "SNO")
    print(answer)


    epscom_set("PWR ON")