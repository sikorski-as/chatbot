import pathlib
import shutil

_mkdirs = [
    'setups'
]

_bin_files = [
    'requirements.txt',
    'smartchat.py',
    'params.py',
    'utils.py',
    'data.py',
    'conversation.py',
    'generate_multiple_answers.py',
]

_bin_dirs = [

]

_files = [
    'install.sh',
    'run.sh',
    'install.bat',
    'run.bat',
]

_build_dir = 'chatbot-build'


def main():
    answer = input('Directory "build" will be removed (if exists) with all of its content. y/n?')
    if answer == 'y':
        # clear and create build directory
        shutil.rmtree(_build_dir, ignore_errors=True)

        # create build/ directory
        build_path = pathlib.Path(_build_dir)
        build_path.mkdir(parents=True, exist_ok=True)

        # create build/bin/ directory
        build_bin_path = build_path / 'bin'
        build_bin_path.mkdir(parents=True, exist_ok=True)

        # create optional dirs
        for dir in _mkdirs:
            new_dir = build_path / dir
            new_dir.mkdir(parents=True, exist_ok=True)

        # copy files to build/bin/
        for file in _bin_files:
            destination = build_bin_path / file
            shutil.copy(file, destination)

        # copy files to build/
        for file in _files:
            destination = build_path / file
            shutil.copy(file, destination)

        # copy dirs to build/bin/
        for dir in _bin_dirs:
            destination = build_bin_path / dir
            shutil.copytree(dir, destination)
    else:
        print('aborted')


if __name__ == '__main__':
    main()
