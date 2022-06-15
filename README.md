# setup
1. Exec below commands

Install python3.9.13 in Amazon linux
```
# pyenv
$ git clone https://github.com/pyenv/pyenv.git ~/.pyenv
$ echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
$ echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
$ echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile
$ source ~/.bash_profile

# pyenv-virtualenv
$ git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
$ echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bash_profile
$ source ~/.bash_profile

# install necessary libraries
$ sudo yum install gcc zlib-devel bzip2 bzip2-devel readline readline-devel sqlite sqlite-devel openssl openssl-devel libffi-devel -y

$ pyenv install 3.9.13
$ pyenv versions #確認コマンド

$ pyenv virtualenv 3.9.13 loopholescanner
$ pyenv local loopholescanner

$ pip install -r requirements.txt

# Measure for ImportError: libGL.so.1: cannot open shared object file: No such file or directory
sudo yum install -y mesa-libGL.x86_64
```

2. Put Google Vision API key in data/key
3. Change GOOGLE_KEY in src/loopholeScanner.py

# Port and Theme Setting

1. add config.toml
```
mkdir src/.streamlit

echo '''
[browser]
gatherUsageStats = false
serverAddress = "<Domain>"

[server]
port = 8501

[theme]
base="light"
''' > src/.streamlit/config.toml
```

2. port forwarding
`sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8501`

# run

```
cd src
streamlit run main.py
```
