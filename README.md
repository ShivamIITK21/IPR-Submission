## IPR-Submission

### How to Run

- Installing Dependencies

```
pip install -r requirements.txt
```

- Configure the params in train.py, such as device and batch size
- Run the train script for training
```
python train.py
```

- View the logs in tensorboard
```
tensorboard --logdir=./tensorboard_logs
```

- To run the linear probe test, specify the saved model path in lin_probe.py and run the script

```
python lin_probe.py
```

