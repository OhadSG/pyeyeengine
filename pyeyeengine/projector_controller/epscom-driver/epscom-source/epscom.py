import subprocess

EPSCOM_CMD = 'epscom-cmd'

def full_command(cmd):
    return "./%s \"%s\"" %(EPSCOM_CMD, cmd)

def epscom_set(cmd):
    run_comand_safe(full_command(cmd))

def epscom_get(cmd):
    answer = run_comand_safe(full_command(cmd))
    return clean_returned_byte_string(answer)

def clean_returned_byte_string(answer):
    return answer.decode("utf-8").replace("\n", "").split("=")

def run_comand_safe(command):
    try:
        answer = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        return answer
    except subprocess.CalledProcessError as e:
        return e.output
        # raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

if __name__ == '__main__':
    answer = epscom_get("LAMP?")
    assert(answer[0]=="LAMP")
    print(answer)

    answer = epscom_get("PWR?")
    assert(answer[0]=="PWR")
    print(answer)


    epscom_set("PWR ON")