# setup

1. Exec below commands

```
python3 -m venv env 
source env/bin/activate
pip install -r requirements.txt
```

2. Put Google Vision API key in data/key
3. Change GOOGLE_KEY in src/loopholeScanner.py

# run

```
cd src
streamlit run main.py
```

choose an image from `data/img/test/`

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

