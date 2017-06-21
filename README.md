# LiTeFlow

**LiTeFlow** is a **Li**ghtweight **Te**nsor**Flow** extension library. It collects a bunch
of functions that can help you building and testing some complex neural network layers.


## DISCLAIMER

LiTeFlow is not intended to be a super-layer like Keras
or Lasagne: it's just a collection of handy off-the-shelf code snippets that can
make us save some time when implementing some common operations. As soon as the 
official TensorFlow project will be provided with higher level APIs, thisl library 
will become useless -- I know, with such a cool name it's kind of a waste, but building
something more is way beyond the scope of my work and my abilities. Said that: have fun!


## Installation.

Do you run a GPU or CPU? Whatever, you need to install the proper tensorflow version
before installing `LiTeFlow`. To use `LiTeFlow`, you can install it via `pip` running:

    $ pip install git+https://github.com/petrux/LiTeFlow.git@master


## Development

If you wan to contribute to `LiTeFlow` or just dig a bit deeper into it, clone the repository:

    $ git clone https://github.com/petrux/LiTeFlow.git

then get into the directory and run one of the script in the `bin` directory to create
a virtual environent to work with:

    $ cd LiTeFlow
    $ ./bin/py3venv.sh

We **strongly** recommend to use Python 3 and to run a virtual environment. 
The `./bin/py3venv.sh` script will create a `.py3venv` directory with the virtual
environment. Now, you just need to activate it and install the proper version of
`TensorFlow` for your machine (GPU or CPU).

    $ source .py3venv/bin/activate
    $ pip install tensorflow-gpu