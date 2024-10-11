import tempfile
import subprocess
import os
import requests

DATA_DIR = "netlib/"
DATA_NAMES = [
    "25fv47",
    "80bau3b",
    "adlittle",
    "afiro",
    "agg",
    "agg2",
    "agg3",
    "bandm",
    "beaconfd",
    "blend",
    "bnl1",
    "bnl2",
    "boeing1",
    "boeing2",
    "bore3d",
    "brandy",
    "capri",
    "cycle",
    "czprob",
    "d2q06c",
    "d6cube",
    "degen2",
    "degen3",
    "dfl001",
    "e226",
    "etamacro",
    "fffff800",
    "finnis",
    "fit1d",
    "fit1p",
    "fit2d",
    "fit2p",
    "forplan",
    "ganges",
    "gfrd-pnc",
    "greenbea",
    "greenbeb",
    "grow15",
    "grow22",
    "grow7",
    "israel",
    "kb2",
    "lotfi",
    "maros",
    "maros-r7",
    "modszk1",
    "nesm",
    "perold",
    "pilot",
    "pilot.ja",
    "pilot.we",
    "pilot4",
    "pilot87",
    "pilotnov",
    "recipe",
    "sc105",
    "sc205",
    "sc50a",
    "sc50b",
    "scagr25",
    "scagr7",
    "scfxm1",
    "scfxm2",
    "scfxm3",
    "scorpion",
    "scrs8",
    "scsd1",
    "scsd6",
    "scsd8",
    "sctap1",
    "sctap2",
    "sctap3",
    "seba",
    "share1b",
    "share2b",
    "shell",
    "ship04l",
    "ship04s",
    "ship08l",
    "ship08s",
    "ship12l",
    "ship12s",
    "sierra",
    "stair",
    "standata",
    "standgub",
    "standmps",
    "stocfor1",
    "stocfor2",
    "stocfor3",
    "truss",
    "tuff",
    "vtp.base",
    "wood1p",
    "woodw",
]


def download(file_name):
    """
    Download a file from netlib
    """

    url = "https://www.netlib.org/lp/data/" + file_name
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as error:
        print(f"Request to {url} failed: {error}")

    return response.text


def get_decode_program():
    """
    Download and compile C program to decode netlib data
    """

    code = download("emps.c")
    bin = tempfile.NamedTemporaryFile(suffix="")

    with tempfile.NamedTemporaryFile(suffix=".c") as src:
        src.write(code.encode())
        compile_command = ["gcc", src.name, "-o", bin.name, "-O3", "-w"]
        subprocess.run(compile_command, check=True)

    return bin


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    with get_decode_program() as bin:
        for data_name in DATA_NAMES:
            print(f"{data_name}... ", end="")
            data = download(data_name)
            with open(DATA_DIR + data_name + ".mps", "w") as f:
                try:
                    subprocess.run(
                        bin.name, input=data, text=True, stdout=f, check=True
                    )
                    print("OK")
                except subprocess.CalledProcessError:
                    print("Decode failed!")


if __name__ == "__main__":
    main()
