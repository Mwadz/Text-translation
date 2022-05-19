{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Mwadz/Text-translation/blob/main/Final_App_Translation.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "nV1hwDh9s1ZP"
      },
      "source": [
        "##  Importing the libraries"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "m7PCjt709-91",
        "outputId": "e177d70e-bea8-4c1b-846d-6f76a6c0a361"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "2.8.0\n"
          ]
        }
      ],
      "source": [
        "from __future__ import absolute_import, division, print_function\n",
        "# Import TensorFlow >= 1.10 and enable eager execution\n",
        "py -m pip3 install tensorflow\n",
        "import tensorflow as tf\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import matplotlib.ticker as ticker\n",
        "from sklearn.model_selection import train_test_split\n",
        "import unicodedata\n",
        "import re\n",
        "import numpy as np\n",
        "import os\n",
        "import time\n",
        "print(tf.__version__)#to check the tensorflow version"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "kY6rStF8huvw"
      },
      "source": [
        "##  Shapechecker"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tTMhMqrChhVV"
      },
      "source": [
        "Function to prevent loading of data of wrong shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "RCF51Lophw-D"
      },
      "outputs": [],
      "source": [
        "class ShapeChecker():\n",
        "  def __init__(self):\n",
        "    # Keep a cache of every axis-name seen\n",
        "    self.shapes = {}\n",
        "\n",
        "  def __call__(self, tensor, names, broadcast=False):\n",
        "    if not tf.executing_eagerly():\n",
        "      return\n",
        "\n",
        "    if isinstance(names, str):\n",
        "      names = (names,)\n",
        "\n",
        "    shape = tf.shape(tensor)\n",
        "    rank = tf.rank(tensor)\n",
        "\n",
        "    if rank != len(names):\n",
        "      raise ValueError(f'Rank mismatch:\\n'\n",
        "                       f'    found {rank}: {shape.numpy()}\\n'\n",
        "                       f'    expected {len(names)}: {names}\\n')\n",
        "\n",
        "    for i, name in enumerate(names):\n",
        "      if isinstance(name, int):\n",
        "        old_dim = name\n",
        "      else:\n",
        "        old_dim = self.shapes.get(name, None)\n",
        "      new_dim = shape[i]\n",
        "\n",
        "      if (broadcast and new_dim == 1):\n",
        "        continue\n",
        "\n",
        "      if old_dim is None:\n",
        "        # If the axis name is new, add its length to the cache.\n",
        "        self.shapes[name] = new_dim\n",
        "        continue\n",
        "\n",
        "      if new_dim != old_dim:\n",
        "        raise ValueError(f\"Shape mismatch for dimension: '{name}'\\n\"\n",
        "                         f\"    found: {new_dim}\\n\"\n",
        "                         f\"    expected: {old_dim}\\n\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_7oYNhBitA8V"
      },
      "source": [
        "##  Loading the datasets"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "C6rq-9h7-LVk"
      },
      "outputs": [],
      "source": [
        "# Loading the datasets\n",
        "english = pd.read_csv('/content/english.txt', sep='delimiter', engine = 'python', header=None)\n",
        "kiuk = pd.read_csv('/content/Kikuyu.txt', sep='delimiter', engine = 'python', header=None)\n",
        "kale = pd.read_csv('/content/kale.txt', sep='delimiter',  engine = 'python', header=None)\n",
        "luo = pd.read_csv('/content/luo.txt', sep='delimiter', engine = 'python', header=None)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "OXmwhiAZtO7C"
      },
      "source": [
        "##  Previewing the datasets"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "O6LnmJjnt3PE",
        "outputId": "4a8f4333-8ef8-4591-a7ef-3edb1fc1c9bf"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "The dataset has 176 rows and 1 columns\n",
            "The dataset has 176 rows and 1 columns\n",
            "The dataset has 176 rows and 1 columns\n",
            "The dataset has 176 rows and 1 columns\n"
          ]
        }
      ],
      "source": [
        "# print the shape of the various datasets\n",
        "files = [english, kiuk, luo, kale]\n",
        "dataset_names = ['English', 'Kikuyu', 'Luo', 'Kalenjin']\n",
        "for file in files:\n",
        "  #for index in range(len(dataset_names)):\n",
        "    rows, columns = file.shape\n",
        "    print(f'The dataset has {rows} rows and {columns} columns')\n",
        "    "
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ItSP4diOv_-v"
      },
      "source": [
        "##  Pre_processing"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "23vUJSRhxhsn"
      },
      "outputs": [],
      "source": [
        "# (Unicode is the universal character encoding used to process, store and facilitate the interchange of text data in any language \n",
        "# while ASCII is used for the representation of text such as symbols, letters, digits, etc)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "wKESEW4qxFGo"
      },
      "source": [
        "Preprocessing steps includes\n",
        "\n",
        "- Converting the unicode file to ascii\n",
        "- Creating a space between a word and the punctuation following it\n",
        "eg: “he is a boy.” => “he is a boy .” Reference\n",
        "- Replacing everything with space except (a-z, A-Z, “.”, “?”, “!”, “,”)\n",
        "- Adding a start and an end token to the sentence so that the model know when to start and stop predicting.\n",
        "- Removing the accents\n",
        "- Cleaning the sentences\n",
        "- Return word pairs in the format: [ENGLISH, LUO]\n",
        "- Creating a word -> index mapping (e.g,. 'Further' -> 5) and vice-versa. (e.g., 5 -> 'Further' ) for each language."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9Ns6OegFPV-j"
      },
      "outputs": [],
      "source": [
        "# Creating an index column for Kalenjin file\n",
        "kale['index_col'] = kale.index"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "-LYMqkFVw8Oe"
      },
      "outputs": [],
      "source": [
        "# Creating an index column for Kalenjin file\n",
        "english['index_col'] = kale.index"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "vvDFhp9CObTv"
      },
      "outputs": [],
      "source": [
        "# Joining the English and Kalenjin file with the Index column\n",
        "df_kale = pd.merge(english, kale, on = 'index_col')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "LnP8rk38PtsD"
      },
      "outputs": [],
      "source": [
        "# Renaming the Kalenjin Columns\n",
        "df_kale.head()\n",
        "df_kale.columns = ['feature', 'index', 'target']"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "UhVBZIzUQow5"
      },
      "outputs": [],
      "source": [
        "# Dropping the Index column in the Kalenjjin file\n",
        "df_kale.columns\n",
        "df_kale = df_kale.drop(columns = ['index'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "CIEp-CbeQypJ",
        "outputId": "e1de789c-b885-4513-cea4-bd02ed39575d"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-87600fc8-3d2b-47ef-b019-eb8a0b8b363b\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>feature</th>\n",
              "      <th>target</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Blessed are the undefiled in the way, who walk...</td>\n",
              "      <td>Boiboen che igesunotgei eng’ oret, Che bendote...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2 Blessed are they that keep his testimonies, ...</td>\n",
              "      <td>Boiboen ichek che ribei baornatosiekyik, Che c...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3 They also do no iniquity: they walk in his w...</td>\n",
              "      <td>Ee, mayaei ichek che ma bo iman; Bendote ortin...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4 Thou hast commanded us to keep thy precepts ...</td>\n",
              "      <td>Kiing’at konetisiosieguk, Ile kisub eng’ kagii...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5 O that my ways were directed to keep thy sta...</td>\n",
              "      <td>Ee, kata mie nda ka kimen ortinwekyuk Si kobii...</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-87600fc8-3d2b-47ef-b019-eb8a0b8b363b')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-87600fc8-3d2b-47ef-b019-eb8a0b8b363b button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-87600fc8-3d2b-47ef-b019-eb8a0b8b363b');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "                                             feature  \\\n",
              "0  Blessed are the undefiled in the way, who walk...   \n",
              "1  2 Blessed are they that keep his testimonies, ...   \n",
              "2  3 They also do no iniquity: they walk in his w...   \n",
              "3  4 Thou hast commanded us to keep thy precepts ...   \n",
              "4  5 O that my ways were directed to keep thy sta...   \n",
              "\n",
              "                                              target  \n",
              "0  Boiboen che igesunotgei eng’ oret, Che bendote...  \n",
              "1  Boiboen ichek che ribei baornatosiekyik, Che c...  \n",
              "2  Ee, mayaei ichek che ma bo iman; Bendote ortin...  \n",
              "3  Kiing’at konetisiosieguk, Ile kisub eng’ kagii...  \n",
              "4  Ee, kata mie nda ka kimen ortinwekyuk Si kobii...  "
            ]
          },
          "execution_count": 12,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Displaying the first rows on the Kalenjin file\n",
        "df_kale.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 261
        },
        "id": "pi7fEmJ4RKrg",
        "outputId": "c2087152-2988-47f2-9b9a-beda53d297c3"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:2: FutureWarning: The default value of regex will change from True to False in a future version.\n",
            "  \n"
          ]
        },
        {
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-b510ee87-2c5e-451d-a6cc-35af535b3d11\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>feature</th>\n",
              "      <th>target</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Blessed are the undefiled in the way, who walk...</td>\n",
              "      <td>Boiboen che igesunotgei eng’ oret, Che bendote...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Blessed are they that keep his testimonies, a...</td>\n",
              "      <td>Boiboen ichek che ribei baornatosiekyik, Che c...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>They also do no iniquity: they walk in his ways.</td>\n",
              "      <td>Ee, mayaei ichek che ma bo iman; Bendote ortin...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Thou hast commanded us to keep thy precepts d...</td>\n",
              "      <td>Kiing’at konetisiosieguk, Ile kisub eng’ kagii...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>O that my ways were directed to keep thy stat...</td>\n",
              "      <td>Ee, kata mie nda ka kimen ortinwekyuk Si kobii...</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-b510ee87-2c5e-451d-a6cc-35af535b3d11')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-b510ee87-2c5e-451d-a6cc-35af535b3d11 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-b510ee87-2c5e-451d-a6cc-35af535b3d11');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "                                             feature  \\\n",
              "0  Blessed are the undefiled in the way, who walk...   \n",
              "1   Blessed are they that keep his testimonies, a...   \n",
              "2   They also do no iniquity: they walk in his ways.   \n",
              "3   Thou hast commanded us to keep thy precepts d...   \n",
              "4   O that my ways were directed to keep thy stat...   \n",
              "\n",
              "                                              target  \n",
              "0  Boiboen che igesunotgei eng’ oret, Che bendote...  \n",
              "1  Boiboen ichek che ribei baornatosiekyik, Che c...  \n",
              "2  Ee, mayaei ichek che ma bo iman; Bendote ortin...  \n",
              "3  Kiing’at konetisiosieguk, Ile kisub eng’ kagii...  \n",
              "4  Ee, kata mie nda ka kimen ortinwekyuk Si kobii...  "
            ]
          },
          "execution_count": 13,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Removing the numbers at the beginning of the feature column\n",
        "df_kale['feature'] = df_kale['feature'].str.replace('\\d+', '')\n",
        "\n",
        "df_kale.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 513
        },
        "id": "JxzDPMwJc3yR",
        "outputId": "467181ec-c38b-4e34-915a-5e2364dc9f10"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:1: FutureWarning: The default value of regex will change from True to False in a future version.\n",
            "  \"\"\"Entry point for launching an IPython kernel.\n",
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:3: FutureWarning: The default value of regex will change from True to False in a future version.\n",
            "  This is separate from the ipykernel package so we can avoid doing imports until\n"
          ]
        },
        {
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-77fa554d-d56a-489f-83c5-fda23e17c9c2\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>feature</th>\n",
              "      <th>target</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Blessed are the undefiled in the way, who walk...</td>\n",
              "      <td>Boiboen che igesunotgei eng’ oret, Che bendote...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Blessed are they that keep his testimonies, a...</td>\n",
              "      <td>Boiboen ichek che ribei baornatosiekyik, Che c...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>They also do no iniquity: they walk in his ways.</td>\n",
              "      <td>Ee, mayaei ichek che ma bo iman; Bendote ortin...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Thou hast commanded us to keep thy precepts d...</td>\n",
              "      <td>Kiing’at konetisiosieguk, Ile kisub eng’ kagii...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>O that my ways were directed to keep thy stat...</td>\n",
              "      <td>Ee, kata mie nda ka kimen ortinwekyuk Si kobii...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>171</th>\n",
              "      <td>My tongue shall speak of thy word: for all th...</td>\n",
              "      <td>Ingotien ng’elyeptanyu agobo ng’olyondeng’ung’...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>172</th>\n",
              "      <td>Let thine hand help me; for I have chosen thy...</td>\n",
              "      <td>Ingochobok eung’ung’ kotoreta; Amu kialewen ko...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>173</th>\n",
              "      <td>I have longed for thy salvation, O Lord; and ...</td>\n",
              "      <td>Kigoama emosto agobo yetuneng’ung’, ee Jehovah...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>174</th>\n",
              "      <td>Let my soul live, and it shall praise thee; a...</td>\n",
              "      <td>Ingosob sobondanyu, si kolosun; Ak ingotoreta ...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>175</th>\n",
              "      <td>I have gone astray like a lost sheep; seek th...</td>\n",
              "      <td>Kiabetote ko u kechiriet ne betot; cheng’ kibo...</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>176 rows × 2 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-77fa554d-d56a-489f-83c5-fda23e17c9c2')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-77fa554d-d56a-489f-83c5-fda23e17c9c2 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-77fa554d-d56a-489f-83c5-fda23e17c9c2');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "                                               feature  \\\n",
              "0    Blessed are the undefiled in the way, who walk...   \n",
              "1     Blessed are they that keep his testimonies, a...   \n",
              "2     They also do no iniquity: they walk in his ways.   \n",
              "3     Thou hast commanded us to keep thy precepts d...   \n",
              "4     O that my ways were directed to keep thy stat...   \n",
              "..                                                 ...   \n",
              "171   My tongue shall speak of thy word: for all th...   \n",
              "172   Let thine hand help me; for I have chosen thy...   \n",
              "173   I have longed for thy salvation, O Lord; and ...   \n",
              "174   Let my soul live, and it shall praise thee; a...   \n",
              "175   I have gone astray like a lost sheep; seek th...   \n",
              "\n",
              "                                                target  \n",
              "0    Boiboen che igesunotgei eng’ oret, Che bendote...  \n",
              "1    Boiboen ichek che ribei baornatosiekyik, Che c...  \n",
              "2    Ee, mayaei ichek che ma bo iman; Bendote ortin...  \n",
              "3    Kiing’at konetisiosieguk, Ile kisub eng’ kagii...  \n",
              "4    Ee, kata mie nda ka kimen ortinwekyuk Si kobii...  \n",
              "..                                                 ...  \n",
              "171  Ingotien ng’elyeptanyu agobo ng’olyondeng’ung’...  \n",
              "172  Ingochobok eung’ung’ kotoreta; Amu kialewen ko...  \n",
              "173  Kigoama emosto agobo yetuneng’ung’, ee Jehovah...  \n",
              "174  Ingosob sobondanyu, si kolosun; Ak ingotoreta ...  \n",
              "175  Kiabetote ko u kechiriet ne betot; cheng’ kibo...  \n",
              "\n",
              "[176 rows x 2 columns]"
            ]
          },
          "execution_count": 14,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "df_kale['feature'] = df_kale['feature'].str.replace('\\d+', '')\n",
        "\n",
        "df_kale['target'] = df_kale['target'].str.replace('\\d+', '')\n",
        "\n",
        "df_kale"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "5bv0qJn7aYot"
      },
      "outputs": [],
      "source": [
        "# from google.colab import files\n",
        "# files.download(\"df_kale.csv\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "npB5UQG4epUy"
      },
      "outputs": [],
      "source": [
        "inp = df_kale['target'].to_list()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "5oa6ivQBcE4w"
      },
      "outputs": [],
      "source": [
        "targ = df_kale['feature'].to_list()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tPz-5B-ve5OG"
      },
      "source": [
        "##  Creating tf_dataset"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "skpuhugPjU6G"
      },
      "source": [
        "Creating a tf.data.Dataset of strings that shuffles and batches them efficiently:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Yn3o0fzUez0p",
        "outputId": "acb2d335-c0f8-49f3-ab4b-10c2a9198f73"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "2022-05-19 03:43:03.132906: E tensorflow/stream_executor/cuda/cuda_driver.cc:271] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected\n"
          ]
        },
        {
          "data": {
            "text/plain": [
              "<BatchDataset element_spec=(TensorSpec(shape=(None,), dtype=tf.string, name=None), TensorSpec(shape=(None,), dtype=tf.string, name=None))>"
            ]
          },
          "execution_count": 18,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Tells TensorFlow to create a buffer of at most buffer_size elements, and a background thread to fill that buffer in the background\n",
        "BUFFER_SIZE = len(inp)\n",
        "\n",
        "# Number of samples to be feed into the neural network\n",
        "BATCH_SIZE = 3\n",
        "\n",
        "# Creating the dataset and shuffling it \n",
        "dataset = tf.data.Dataset.from_tensor_slices((inp, targ)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)\n",
        "dataset "
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jX98Wa1bHlsE",
        "outputId": "7485f1d5-c997-4837-cbeb-1d0b605dfb6f"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "tf.Tensor(\n",
            "[b'Asain ale taach konunoikab kutinnyu, ee Jehovah, Ak ineta kiruogutiguk.'\n",
            " b'Ingochobok eung\\xe2\\x80\\x99ung\\xe2\\x80\\x99 kotoreta; Amu kialewen konetisiosieguk.'\n",
            " b'Kiatinye ko u noto, Amu kiarib konetisiosieguk.'], shape=(3,), dtype=string)\n",
            "\n",
            "tf.Tensor(\n",
            "[b' Accept, I beseech thee, the freewill offerings of my mouth, O Lord, and teach me thy judgments.'\n",
            " b' Let thine hand help me; for I have chosen thy precepts.'\n",
            " b' This I had, because I kept thy precepts.'], shape=(3,), dtype=string)\n"
          ]
        }
      ],
      "source": [
        "for example_input_batch, example_target_batch in dataset.take(1):\n",
        "  print(example_input_batch[:5])\n",
        "  print()\n",
        "  print(example_target_batch[:5])\n",
        "  break"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_kCgCCkofZql"
      },
      "source": [
        "##  Text processing"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "QxJjvOzYfeSA"
      },
      "source": [
        "### i) Standardization"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zbsccb0GlpPP"
      },
      "source": [
        "Since the model is dealing with multilingual text with a limited vocabulary standardization of the text is crucial. Steps;\n",
        "1.  Unicode normalization to split accented characters\n",
        "2.  replace compatibility characters with their ASCII equivalents."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Ncfk-gznZ78G"
      },
      "outputs": [],
      "source": [
        ""
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "qimZmadJge4G"
      },
      "outputs": [],
      "source": [
        "import tensorflow_text as tf_text"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aaXEDlsSfcDu",
        "outputId": "ce34b7d7-a67d-4833-b0ff-2a008bafceed"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "b'Kiacheng\\xe2\\x80\\x99in eng\\xe2\\x80\\x99 muguleldanyu tugul'\n",
            "b'Kiacheng\\xe2\\x80\\x99in eng\\xe2\\x80\\x99 muguleldanyu tugul'\n"
          ]
        }
      ],
      "source": [
        "# example of a text normalized and uni encoded\n",
        "sample_text = tf.constant('Kiacheng’in eng’ muguleldanyu tugul')\n",
        "\n",
        "print(sample_text.numpy())\n",
        "print(tf_text.normalize_utf8(sample_text, 'NFKD').numpy())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "hvbI7A7hgaBj"
      },
      "outputs": [],
      "source": [
        "# Unicode normalization \n",
        "def tf_lower_and_split_punct(text):\n",
        "  # Split accecented characters.\n",
        "  text = tf_text.normalize_utf8(text, 'NFKD')\n",
        "  text = tf.strings.lower(text)\n",
        "  # Keep space, a to z, and select punctuation.\n",
        "  text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')\n",
        "  # Add spaces around punctuation.\n",
        "  text = tf.strings.regex_replace(text, '[.?!,¿]', r' \\0 ')\n",
        "  # Strip whitespace.\n",
        "  text = tf.strings.strip(text)\n",
        "\n",
        "  text = tf.strings.join(['[START]', text, '[END]'], separator=' ')\n",
        "  return text"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zTC9ec0vg2w9",
        "outputId": "9d4bc4b5-34a0-4252-b659-39cdb311c040"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Kiacheng’in eng’ muguleldanyu tugul\n",
            "[START] kiachengin eng muguleldanyu tugul [END]\n"
          ]
        }
      ],
      "source": [
        "# Priniting an example of the original text\n",
        "print(sample_text.numpy().decode())\n",
        "\n",
        "# printing the text afterunicode normalization\n",
        "print(tf_lower_and_split_punct(sample_text).numpy().decode())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "pcG-ObPshAmb"
      },
      "outputs": [],
      "source": [
        "# Extracting and coverting input text to sequences of tokens\n",
        "# max_vocab_size limit RAM usage during the initial scan of the training corpus to discover the vocabulary.\n",
        "max_vocab_size = 25000 \n",
        "\n",
        "input_text_processor = tf.keras.layers.TextVectorization(\n",
        "    standardize=tf_lower_and_split_punct,\n",
        "    max_tokens=max_vocab_size)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "euDjRG9ZhEPH",
        "outputId": "1501d2df-7fb7-4ac4-e1b4-996b16bae4eb"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "['', '[UNK]', '[START]', '[END]', '.', ',', 'ak', 'eng', 'amu', 'ngatutiguk']"
            ]
          },
          "execution_count": 25,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Reading one epoch of the training data with the adapt method \n",
        "input_text_processor.adapt(inp)\n",
        "\n",
        "# Here are the first 10 words from the vocabulary:\n",
        "input_text_processor.get_vocabulary()[:10]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YdrMYNIzhKO1",
        "outputId": "bfb06f11-ab85-45f2-d722-791199f7c3fb"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "['', '[UNK]', 'thy', '[START]', '[END]', '.', 'i', ',', 'me', 'and']"
            ]
          },
          "execution_count": 26,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Using the Kalenjin TextVectorization layer to build the English layer with .adapt() method\n",
        "output_text_processor = tf.keras.layers.TextVectorization(\n",
        "    standardize=tf_lower_and_split_punct,\n",
        "    max_tokens=max_vocab_size)\n",
        "\n",
        "output_text_processor.adapt(targ)\n",
        "output_text_processor.get_vocabulary()[:10]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tGz0sQG2hRI9",
        "outputId": "f13d8c06-9855-4f9c-8b8b-3c84c1517555"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "<tf.Tensor: shape=(3, 10), dtype=int64, numpy=\n",
              "array([[  2, 176,  36, 207, 278,  49,   5,  15,  18,   5],\n",
              "       [  2, 411, 428, 271,   8, 149,  21,   4,   3,   0],\n",
              "       [  2, 147,  13,  17,  34,   5,   8,  93,  21,   4]])>"
            ]
          },
          "execution_count": 27,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Using the layers created to convert a batch of strings into a batch of token IDs\n",
        "example_tokens = input_text_processor(example_input_batch)\n",
        "example_tokens[:3, :10]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 298
        },
        "id": "AMFb-oOlhSBZ",
        "outputId": "d5efbb86-8f65-4ba9-9c68-440e8a881695"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "Text(0.5, 1.0, 'Mask')"
            ]
          },
          "execution_count": 28,
          "metadata": {},
          "output_type": "execute_result"
        },
        {
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 2 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "# Applying the token IDs that are zero-padded that can be turned into a mask\n",
        "plt.subplot(1, 2, 1)\n",
        "plt.pcolormesh(example_tokens)\n",
        "plt.title('Token IDs')\n",
        "\n",
        "plt.subplot(1, 2, 2)\n",
        "plt.pcolormesh(example_tokens != 0)\n",
        "plt.title('Mask')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "W34FhUHQhXYB"
      },
      "outputs": [],
      "source": [
        "# Defining constants for the model\n",
        "# Embedding layer enables us to convert each word into a fixed length vector of defined size\n",
        "embedding_dim = 512\n",
        "units = 1024"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "yfs8q1paheh8"
      },
      "source": [
        "##  The encoder"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "MfGvULMIxdyM"
      },
      "source": [
        "The first thing to do is build the encoder. The process is as follows:\n",
        "\n",
        "1. Taking a list of token IDs. \n",
        "\n",
        "2. Using the embedding vector for each token.\n",
        "\n",
        "3. Processessing the embeddings into a new sequence "
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "JF5tj3-LhaoV"
      },
      "outputs": [],
      "source": [
        "# Applying the  list of token IDs\n",
        "class Encoder(tf.keras.layers.Layer):\n",
        "  def __init__(self, input_vocab_size, embedding_dim, enc_units):\n",
        "    super(Encoder, self).__init__()\n",
        "    self.enc_units = enc_units\n",
        "    self.input_vocab_size = input_vocab_size\n",
        "\n",
        "    # The embedding layer converts tokens to vectors\n",
        "    self.embedding = tf.keras.layers.Embedding(self.input_vocab_size,\n",
        "                                               embedding_dim)\n",
        "\n",
        "    # The GRU RNN layer processes those vectors sequentially.\n",
        "    self.gru = tf.keras.layers.GRU(self.enc_units,\n",
        "                                   # Return the sequence and state\n",
        "                                   return_sequences=True,\n",
        "                                   return_state=True,\n",
        "                                   recurrent_initializer='glorot_uniform')\n",
        "\n",
        "  def call(self, tokens, state=None):\n",
        "    shape_checker = ShapeChecker()\n",
        "    shape_checker(tokens, ('batch', 's'))\n",
        "\n",
        "    # 2. The embedding layer looks up the embedding for each token.\n",
        "    vectors = self.embedding(tokens)\n",
        "    shape_checker(vectors, ('batch', 's', 'embed_dim'))\n",
        "\n",
        "    # 3. The GRU processes the embedding sequence.\n",
        "    #    output shape: (batch, s, enc_units)\n",
        "    #    state shape: (batch, enc_units)\n",
        "    output, state = self.gru(vectors, initial_state=state)\n",
        "    shape_checker(output, ('batch', 's', 'enc_units'))\n",
        "    shape_checker(state, ('batch', 'enc_units'))\n",
        "\n",
        "    # 4. Returns the new sequence and its state.\n",
        "    return output, state"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OCGAOYOKhhyU",
        "outputId": "36e0e95d-cf80-4941-d059-6ca92a479a52"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Input batch, shape (batch): (3,)\n",
            "Input batch tokens, shape (batch, s): (3, 15)\n",
            "Encoder output, shape (batch, s, units): (3, 15, 1024)\n",
            "Encoder state, shape (batch, units): (3, 1024)\n"
          ]
        }
      ],
      "source": [
        "# Convert the input text to tokens.\n",
        "example_tokens = input_text_processor(example_input_batch)\n",
        "\n",
        "# Encode the input sequence.\n",
        "encoder = Encoder(input_text_processor.vocabulary_size(),\n",
        "                  embedding_dim, units)\n",
        "example_enc_output, example_enc_state = encoder(example_tokens)\n",
        "\n",
        "print(f'Input batch, shape (batch): {example_input_batch.shape}')\n",
        "print(f'Input batch tokens, shape (batch, s): {example_tokens.shape}')\n",
        "print(f'Encoder output, shape (batch, s, units): {example_enc_output.shape}')\n",
        "print(f'Encoder state, shape (batch, units): {example_enc_state.shape}')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "86t5JagIh0jA"
      },
      "outputs": [],
      "source": [
        ""
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "VS6yGwmvh7wE"
      },
      "source": [
        "##  The attention head"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "pyENzTlKzQJL"
      },
      "source": [
        "The decoder uses attention to selectively focus on parts of the input sequence. The attention takes a sequence of vectors as input for each example and returns an \"attention\" vector for each example. "
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "CNl8GtARh-_j"
      },
      "outputs": [],
      "source": [
        "# The BahdanauAttention class handles the weight matrices in a pair of dense layers and calls the builtin implementation\n",
        "class BahdanauAttention(tf.keras.layers.Layer):\n",
        "  def __init__(self, units):\n",
        "    super().__init__()\n",
        "    # For Eqn. (4), the  Bahdanau attention\n",
        "    self.W1 = tf.keras.layers.Dense(units, use_bias=False)\n",
        "    self.W2 = tf.keras.layers.Dense(units, use_bias=False)\n",
        "\n",
        "    self.attention = tf.keras.layers.AdditiveAttention()\n",
        "\n",
        "  def call(self, query, value, mask):\n",
        "    shape_checker = ShapeChecker()\n",
        "    shape_checker(query, ('batch', 't', 'query_units'))\n",
        "    shape_checker(value, ('batch', 's', 'value_units'))\n",
        "    shape_checker(mask, ('batch', 's'))\n",
        "\n",
        "    # From Eqn. (4), `W1@ht`.\n",
        "    w1_query = self.W1(query)\n",
        "    shape_checker(w1_query, ('batch', 't', 'attn_units'))\n",
        "\n",
        "    # From Eqn. (4), `W2@hs`.\n",
        "    w2_key = self.W2(value)\n",
        "    shape_checker(w2_key, ('batch', 's', 'attn_units'))\n",
        "\n",
        "    query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)\n",
        "    value_mask = mask\n",
        "\n",
        "    context_vector, attention_weights = self.attention(\n",
        "        inputs = [w1_query, value, w2_key],\n",
        "        mask=[query_mask, value_mask],\n",
        "        return_attention_scores = True,\n",
        "    )\n",
        "    shape_checker(context_vector, ('batch', 't', 'value_units'))\n",
        "    shape_checker(attention_weights, ('batch', 't', 's'))\n",
        "\n",
        "    return context_vector, attention_weights"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "s8lEay96iFaK"
      },
      "source": [
        "### i) Attention head layer"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "p3D4q8LqiDDG"
      },
      "outputs": [],
      "source": [
        "# Creating a BahdanauAttention layer\n",
        "attention_layer = BahdanauAttention(units)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kXZOMPBeiLJT",
        "outputId": "9e1175fe-51f6-43d9-8aea-9986259a670d"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "TensorShape([3, 15])"
            ]
          },
          "execution_count": 34,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Excluding the padding\n",
        "(example_tokens != 0).shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cAIY77zQiOLi",
        "outputId": "cd13512a-7864-452b-94b6-614628c28910"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Attention result shape: (batch_size, query_seq_length, units):           (3, 2, 1024)\n",
            "Attention weights shape: (batch_size, query_seq_length, value_seq_length): (3, 2, 15)\n"
          ]
        }
      ],
      "source": [
        "# Later, the decoder will generate this attention query\n",
        "example_attention_query = tf.random.normal(shape=[len(example_tokens), 2, 10])\n",
        "\n",
        "# Attend to the encoded tokens\n",
        "\n",
        "context_vector, attention_weights = attention_layer(\n",
        "    query=example_attention_query,\n",
        "    value=example_enc_output,\n",
        "    mask=(example_tokens != 0))\n",
        "\n",
        "print(f'Attention result shape: (batch_size, query_seq_length, units):           {context_vector.shape}')\n",
        "print(f'Attention weights shape: (batch_size, query_seq_length, value_seq_length): {attention_weights.shape}')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 298
        },
        "id": "S3sxTlpbiSbN",
        "outputId": "a715a595-a3f5-4220-f856-f6410ba2e8fa"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "Text(0.5, 1.0, 'Mask')"
            ]
          },
          "execution_count": 36,
          "metadata": {},
          "output_type": "execute_result"
        },
        {
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 2 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "# attention weights across the sequences at t=0\n",
        "# t is used for slicing, for selecting different parts of the data.\n",
        "plt.subplot(1, 2, 1)\n",
        "plt.pcolormesh(attention_weights[:, 0, :])\n",
        "plt.title('Attention weights')\n",
        "\n",
        "plt.subplot(1, 2, 2)\n",
        "plt.pcolormesh(example_tokens != 0)\n",
        "plt.title('Mask')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jtKEg3H-iWHO",
        "outputId": "4ab45f45-5d72-4708-f747-7f7d7ea8d5fd"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "TensorShape([3, 2, 15])"
            ]
          },
          "execution_count": 37,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Displaying the shape of the attention weights\n",
        "attention_weights.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "nLIfNuypiZs1"
      },
      "outputs": [],
      "source": [
        "attention_slice = attention_weights[0, 0].numpy()\n",
        "attention_slice = attention_slice[attention_slice != 0]"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5-EMQsamifCP"
      },
      "source": [
        "### ii) Toogle code"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 425
        },
        "id": "kC-M1cWiihZl",
        "outputId": "afa59c50-9167-4dcf-8e51-f0dc477f07bd"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "[<matplotlib.lines.Line2D at 0x7fe06a175dd0>]"
            ]
          },
          "execution_count": 39,
          "metadata": {},
          "output_type": "execute_result"
        },
        {
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 0 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 864x432 with 2 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "# Plotting attention weights\n",
        "plt.suptitle('Attention weights for one sequence')\n",
        "\n",
        "plt.figure(figsize=(12, 6))\n",
        "a1 = plt.subplot(1, 2, 1)\n",
        "plt.bar(range(len(attention_slice)), attention_slice)\n",
        "# freeze the xlim\n",
        "plt.xlim(plt.xlim())\n",
        "plt.xlabel('Attention weights')\n",
        "\n",
        "a2 = plt.subplot(1, 2, 2)\n",
        "plt.bar(range(len(attention_slice)), attention_slice)\n",
        "plt.xlabel('Attention weights, zoomed')\n",
        "\n",
        "# zoom in\n",
        "top = max(a1.get_ylim())\n",
        "zoom = 0.85*top\n",
        "a2.set_ylim([0.90*top, top])\n",
        "a1.plot(a1.get_xlim(), [zoom, zoom], color='k')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "8aJ1-TaViluf"
      },
      "source": [
        "### iii) The decoder"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "QRld9r7zaAao"
      },
      "source": [
        "The decoder generates predictions for the next output token.\n",
        "1. The decoder receives the complete encoder output.\n",
        "\n",
        "2. It uses an RNN to keep track of what it has generated so far.\n",
        "\n",
        "3. It uses its RNN output as the query to the attention over the encoder's output, producing the context vector.\n",
        "\n",
        "4. It combines the RNN output and the context vector to generate the attention vector.\n",
        "\n",
        "5. It generates logit predictions for the next token based on the attention vector."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "y2w6cOE_ijbo"
      },
      "outputs": [],
      "source": [
        "# Decoder class and its initializer creates all the necessary layers.\n",
        "\n",
        "class Decoder(tf.keras.layers.Layer):\n",
        "  def __init__(self, output_vocab_size, embedding_dim, dec_units):\n",
        "    super(Decoder, self).__init__()\n",
        "    self.dec_units = dec_units\n",
        "    self.output_vocab_size = output_vocab_size\n",
        "    self.embedding_dim = embedding_dim\n",
        "\n",
        "    # The embedding layer convets token IDs to vectors\n",
        "    self.embedding = tf.keras.layers.Embedding(self.output_vocab_size,\n",
        "                                               embedding_dim)\n",
        "\n",
        "    # The RNN keeps track of what's been generated so far.\n",
        "    self.gru = tf.keras.layers.GRU(self.dec_units,\n",
        "                                   return_sequences=True,\n",
        "                                   return_state=True,\n",
        "                                   recurrent_initializer='glorot_uniform')\n",
        "\n",
        "    # The RNN output will be the query for the attention layer.\n",
        "    self.attention = BahdanauAttention(self.dec_units)\n",
        "\n",
        "    #  Eqn. (3): converting `ct` to `at`\n",
        "    self.Wc = tf.keras.layers.Dense(dec_units, activation=tf.math.tanh,\n",
        "                                    use_bias=False)\n",
        "\n",
        "    # This fully connected layer produces the logits for each\n",
        "    # output token.\n",
        "    self.fc = tf.keras.layers.Dense(self.output_vocab_size)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "4BJX5xOzi2Ir"
      },
      "outputs": [],
      "source": [
        "# Importing libraries\n",
        "import typing\n",
        "from typing import Any, Tuple"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "f7aMUkaXisc8"
      },
      "outputs": [],
      "source": [
        "# Applying the call method for this layer which  takes and returns multiple tensors.\n",
        "# Organizing those into simple container classes.\n",
        "class DecoderInput(typing.NamedTuple):\n",
        "  new_tokens: Any\n",
        "  enc_output: Any\n",
        "  mask: Any\n",
        "\n",
        "class DecoderOutput(typing.NamedTuple):\n",
        "  logits: Any\n",
        "  attention_weights: Any"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "hb46LHidjFpW"
      },
      "outputs": [],
      "source": [
        "# Implementing the call method\n",
        "def call(self,\n",
        "         inputs: DecoderInput,\n",
        "         state=None) -> Tuple[DecoderOutput, tf.Tensor]:\n",
        "  shape_checker = ShapeChecker()\n",
        "  shape_checker(inputs.new_tokens, ('batch', 't'))\n",
        "  shape_checker(inputs.enc_output, ('batch', 's', 'enc_units'))\n",
        "  shape_checker(inputs.mask, ('batch', 's'))\n",
        "\n",
        "  if state is not None:\n",
        "    shape_checker(state, ('batch', 'dec_units'))\n",
        "\n",
        "  # Step 1. Lookup the embeddings\n",
        "  vectors = self.embedding(inputs.new_tokens)\n",
        "  shape_checker(vectors, ('batch', 't', 'embedding_dim'))\n",
        "\n",
        "  # Step 2. Process one step with the RNN\n",
        "  rnn_output, state = self.gru(vectors, initial_state=state)\n",
        "\n",
        "  shape_checker(rnn_output, ('batch', 't', 'dec_units'))\n",
        "  shape_checker(state, ('batch', 'dec_units'))\n",
        "\n",
        "  # Step 3. Use the RNN output as the query for the attention over the\n",
        "  # encoder output.\n",
        "  context_vector, attention_weights = self.attention(\n",
        "      query=rnn_output, value=inputs.enc_output, mask=inputs.mask)\n",
        "  shape_checker(context_vector, ('batch', 't', 'dec_units'))\n",
        "  shape_checker(attention_weights, ('batch', 't', 's'))\n",
        "\n",
        "  # Step 4. Eqn. (3): Join the context_vector and rnn_output\n",
        "  #     [ct; ht] shape: (batch t, value_units + query_units)\n",
        "  context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)\n",
        "\n",
        "  # Step 4. Eqn. (3): `at = tanh(Wc@[ct; ht])`\n",
        "  attention_vector = self.Wc(context_and_rnn_output)\n",
        "  shape_checker(attention_vector, ('batch', 't', 'dec_units'))\n",
        "\n",
        "  # Step 5. Generate logit predictions:\n",
        "  logits = self.fc(attention_vector)\n",
        "  shape_checker(logits, ('batch', 't', 'output_vocab_size'))\n",
        "\n",
        "  return DecoderOutput(logits, attention_weights), state"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "zyzKL0jmjnK3"
      },
      "outputs": [],
      "source": [
        "Decoder.call = call"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "PAUFYYjFjJoL"
      },
      "outputs": [],
      "source": [
        "# Implementing  of the decoder \n",
        "decoder = Decoder(output_text_processor.vocabulary_size(),\n",
        "                  embedding_dim, units)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "QOCS-aMZjMex"
      },
      "outputs": [],
      "source": [
        "# Convert the target sequence, and collect the \"[START]\" tokens\n",
        "example_output_tokens = output_text_processor(example_target_batch)\n",
        "\n",
        "start_index = output_text_processor.get_vocabulary().index('[START]')\n",
        "first_token = tf.constant([[start_index]] * example_output_tokens.shape[0])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "G8tAW2qgjPX-",
        "outputId": "c2d8cbb8-7a06-45a4-efa0-0d2cd091bc01"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "logits shape: (batch_size, t, output_vocab_size) (3, 1, 486)\n",
            "state shape: (batch_size, dec_units) (3, 1024)\n"
          ]
        }
      ],
      "source": [
        "# Run the decoder\n",
        "dec_result, dec_state = decoder(\n",
        "    inputs = DecoderInput(new_tokens=first_token,\n",
        "                          enc_output=example_enc_output,\n",
        "                          mask=(example_tokens != 0)),\n",
        "    state = example_enc_state\n",
        ")\n",
        "\n",
        "print(f'logits shape: (batch_size, t, output_vocab_size) {dec_result.logits.shape}')\n",
        "print(f'state shape: (batch_size, dec_units) {dec_state.shape}')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "r0Sr9yCejyGR"
      },
      "outputs": [],
      "source": [
        "# Sampling a token with the logits\n",
        "sampled_token = tf.random.categorical(dec_result.logits[:, 0, :], num_samples=1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "H_36L8TSj3RB",
        "outputId": "96828e61-21b1-4870-cc5f-fdacfcf994f7"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "array([['quickened'],\n",
              "       ['merciful'],\n",
              "       ['exceedingly']], dtype='<U14')"
            ]
          },
          "execution_count": 49,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Decoding the token as the first word of the output\n",
        "vocab = np.array(output_text_processor.get_vocabulary())\n",
        "first_word = vocab[sampled_token.numpy()]\n",
        "first_word[:5]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "5-K4tG9kkEDX"
      },
      "outputs": [],
      "source": [
        "# Applying the same enc_output, mask and sampled tokens as new tokens.\n",
        "\n",
        "dec_result, dec_state = decoder(\n",
        "    DecoderInput(sampled_token,\n",
        "                 example_enc_output,\n",
        "                 mask=(example_tokens != 0)),\n",
        "    state=dec_state)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5OSaBZEQkLex",
        "outputId": "2911e93f-a88e-4d64-9e58-0e2887fbb6fa"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "array([['much'],\n",
              "       ['taken'],\n",
              "       ['morning']], dtype='<U14')"
            ]
          },
          "execution_count": 51,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Generating a second set of logits using the decoder\n",
        "sampled_token = tf.random.categorical(dec_result.logits[:, 0, :], num_samples=1)\n",
        "first_word = vocab[sampled_token.numpy()]\n",
        "first_word[:5]"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-ng_jdDekShC"
      },
      "source": [
        "##  Training"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "L_FDJCl_gkoW"
      },
      "source": [
        "To train the model we'll follow the following steps:\n",
        "\n",
        "1. A loss function and optimizer to perform the optimization.\n",
        "\n",
        "2. A training step function defining how to update the model for each input/target batch.\n",
        "\n",
        "3. A training loop to drive the training and save checkpoints."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "SCjXzI_1k8Du"
      },
      "source": [
        "### i) Define the loss function"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "C2IcOwyKj8At"
      },
      "outputs": [],
      "source": [
        "# Implementing the loss function and optimizer to perform the optimization.\n",
        "class MaskedLoss(tf.keras.losses.Loss):\n",
        "  def __init__(self):\n",
        "    self.name = 'masked_loss'\n",
        "    self.loss = tf.keras.losses.SparseCategoricalCrossentropy(\n",
        "        from_logits=True, reduction='none')\n",
        "\n",
        "  def __call__(self, y_true, y_pred):\n",
        "    shape_checker = ShapeChecker()\n",
        "    shape_checker(y_true, ('batch', 't'))\n",
        "    shape_checker(y_pred, ('batch', 't', 'logits'))\n",
        "\n",
        "    # Calculate the loss for each item in the batch.\n",
        "    loss = self.loss(y_true, y_pred)\n",
        "    shape_checker(loss, ('batch', 't'))\n",
        "\n",
        "    # Mask off the losses on padding.\n",
        "    mask = tf.cast(y_true != 0, tf.float32)\n",
        "    shape_checker(mask, ('batch', 't'))\n",
        "    loss *= mask\n",
        "\n",
        "    # Return the total.\n",
        "    return tf.reduce_sum(loss)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "CaII-2g3lLmO"
      },
      "source": [
        "### ii) Implementing the training step"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "TTsZs60qkcOQ"
      },
      "outputs": [],
      "source": [
        "# Implementing a model class, the training process will be implemented as the train_step method \n",
        "class TrainTranslator(tf.keras.Model):\n",
        "  def __init__(self, embedding_dim, units,\n",
        "               input_text_processor,\n",
        "               output_text_processor, \n",
        "               use_tf_function=True):\n",
        "    super().__init__()\n",
        "    # Build the encoder and decoder\n",
        "    encoder = Encoder(input_text_processor.vocabulary_size(),\n",
        "                      embedding_dim, units)\n",
        "    decoder = Decoder(output_text_processor.vocabulary_size(),\n",
        "                      embedding_dim, units)\n",
        "\n",
        "    self.encoder = encoder\n",
        "    self.decoder = decoder\n",
        "    self.input_text_processor = input_text_processor\n",
        "    self.output_text_processor = output_text_processor\n",
        "    self.use_tf_function = use_tf_function\n",
        "    self.shape_checker = ShapeChecker()\n",
        "\n",
        "  def train_step(self, inputs):\n",
        "    self.shape_checker = ShapeChecker()\n",
        "    if self.use_tf_function:\n",
        "      return self._tf_train_step(inputs)\n",
        "    else:\n",
        "      return self._train_step(inputs)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "yR_m1uF4kfzg"
      },
      "outputs": [],
      "source": [
        "# Getting a batch of input_text, target_text from the tf.data.Dataset.\n",
        "def _preprocess(self, input_text, target_text):\n",
        "  self.shape_checker(input_text, ('batch',))\n",
        "  self.shape_checker(target_text, ('batch',))\n",
        "\n",
        "  # Convert the text to token IDs\n",
        "  input_tokens = self.input_text_processor(input_text)\n",
        "  target_tokens = self.output_text_processor(target_text)\n",
        "  self.shape_checker(input_tokens, ('batch', 's'))\n",
        "  self.shape_checker(target_tokens, ('batch', 't'))\n",
        "\n",
        "  # Convert IDs to masks.\n",
        "  input_mask = input_tokens != 0\n",
        "  self.shape_checker(input_mask, ('batch', 's'))\n",
        "\n",
        "  target_mask = target_tokens != 0\n",
        "  self.shape_checker(target_mask, ('batch', 't'))\n",
        "\n",
        "  return input_tokens, input_mask, target_tokens, target_mask"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "s-oqNm4zlZZg"
      },
      "outputs": [],
      "source": [
        "TrainTranslator._preprocess = _preprocess"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "fyD9YlGlkhiY"
      },
      "outputs": [],
      "source": [
        "# Applying the _train_step method\n",
        "def _train_step(self, inputs):\n",
        "  input_text, target_text = inputs  \n",
        "\n",
        "  (input_tokens, input_mask,\n",
        "   target_tokens, target_mask) = self._preprocess(input_text, target_text)\n",
        "\n",
        "  max_target_length = tf.shape(target_tokens)[1]\n",
        "\n",
        "  with tf.GradientTape() as tape:\n",
        "    # Encode the input\n",
        "    enc_output, enc_state = self.encoder(input_tokens)\n",
        "    self.shape_checker(enc_output, ('batch', 's', 'enc_units'))\n",
        "    self.shape_checker(enc_state, ('batch', 'enc_units'))\n",
        "\n",
        "    # Initialize the decoder's state to the encoder's final state.\n",
        "    # This only works if the encoder and decoder have the same number of\n",
        "    # units.\n",
        "    dec_state = enc_state\n",
        "    loss = tf.constant(0.0)\n",
        "\n",
        "    for t in tf.range(max_target_length-1):\n",
        "      # Pass in two tokens from the target sequence:\n",
        "      # 1. The current input to the decoder.\n",
        "      # 2. The target for the decoder's next prediction.\n",
        "      new_tokens = target_tokens[:, t:t+2]\n",
        "      step_loss, dec_state = self._loop_step(new_tokens, input_mask,\n",
        "                                             enc_output, dec_state)\n",
        "      loss = loss + step_loss\n",
        "\n",
        "    # Average the loss over all non padding tokens.\n",
        "    average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))\n",
        "\n",
        "  # Apply an optimization step\n",
        "  variables = self.trainable_variables \n",
        "  gradients = tape.gradient(average_loss, variables)\n",
        "  self.optimizer.apply_gradients(zip(gradients, variables))\n",
        "\n",
        "  # Return a dict mapping metric names to current value\n",
        "  return {'batch_loss': average_loss}"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "j3mVALlPmbLQ"
      },
      "outputs": [],
      "source": [
        "TrainTranslator._train_step = _train_step"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "dOehrED0md0-"
      },
      "outputs": [],
      "source": [
        "def _loop_step(self, new_tokens, input_mask, enc_output, dec_state):\n",
        "  input_token, target_token = new_tokens[:, 0:1], new_tokens[:, 1:2]\n",
        "\n",
        "  # Run the decoder one step.\n",
        "  decoder_input = DecoderInput(new_tokens=input_token,\n",
        "                               enc_output=enc_output,\n",
        "                               mask=input_mask)\n",
        "\n",
        "  dec_result, dec_state = self.decoder(decoder_input, state=dec_state)\n",
        "  self.shape_checker(dec_result.logits, ('batch', 't1', 'logits'))\n",
        "  self.shape_checker(dec_result.attention_weights, ('batch', 't1', 's'))\n",
        "  self.shape_checker(dec_state, ('batch', 'dec_units'))\n",
        "\n",
        "  # `self.loss` returns the total for non-padded tokens\n",
        "  y = target_token\n",
        "  y_pred = dec_result.logits\n",
        "  step_loss = self.loss(y, y_pred)\n",
        "\n",
        "  return step_loss, dec_state"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "nNBAPnZwmfFf"
      },
      "outputs": [],
      "source": [
        "TrainTranslator._loop_step = _loop_step"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "__19iWQPmiwa"
      },
      "source": [
        "### iii) Test the training step"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "kf7vv-CqmnDy"
      },
      "outputs": [],
      "source": [
        "# Building a TrainTranslator and configuring it for training using the Model.compile method\n",
        "translator = TrainTranslator(\n",
        "    embedding_dim, units,\n",
        "    input_text_processor=input_text_processor,\n",
        "    output_text_processor=output_text_processor,\n",
        "    use_tf_function=False)\n",
        "\n",
        "# Configure the loss and optimizer\n",
        "translator.compile(\n",
        "    optimizer=tf.optimizers.Adam(),\n",
        "    loss=MaskedLoss(),\n",
        ")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "x7YpBQGgmpeq",
        "outputId": "a3fc894a-583f-492d-ac4d-f146887eb878"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "6.186208623900494"
            ]
          },
          "execution_count": 61,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Testing the train_step model\n",
        "np.log(output_text_processor.vocabulary_size())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "65UeaPVZmsKv"
      },
      "outputs": [],
      "source": [
        "# Applying the tf.function-wrapped _tf_train_step, to maximize performance while training\n",
        "@tf.function(input_signature=[[tf.TensorSpec(dtype=tf.string, shape=[None]),\n",
        "                               tf.TensorSpec(dtype=tf.string, shape=[None])]])\n",
        "def _tf_train_step(self, inputs):\n",
        "  return self._train_step(inputs)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "1pHTE-FLmw4s"
      },
      "outputs": [],
      "source": [
        "TrainTranslator._tf_train_step = _tf_train_step"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "IrDSsPrsmzC5"
      },
      "outputs": [],
      "source": [
        "translator.use_tf_function = True"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5_uZLIwbnChM",
        "outputId": "25fbae2b-8ae0-4631-894d-1fb14b51a875"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "{'batch_loss': <tf.Tensor: shape=(), dtype=float32, numpy=5.816431>}"
            ]
          },
          "execution_count": 65,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Tracing the function\n",
        "translator.train_step([example_input_batch, example_target_batch])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TNHqOaK0nDRa",
        "outputId": "007fd0ad-dc47-406e-a667-5891a31a081e"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'batch_loss': <tf.Tensor: shape=(), dtype=float32, numpy=5.662898>}\n",
            "{'batch_loss': <tf.Tensor: shape=(), dtype=float32, numpy=5.4085126>}\n",
            "{'batch_loss': <tf.Tensor: shape=(), dtype=float32, numpy=4.7730308>}\n",
            "{'batch_loss': <tf.Tensor: shape=(), dtype=float32, numpy=3.5588448>}\n",
            "{'batch_loss': <tf.Tensor: shape=(), dtype=float32, numpy=3.0819104>}\n",
            "{'batch_loss': <tf.Tensor: shape=(), dtype=float32, numpy=3.1132476>}\n",
            "{'batch_loss': <tf.Tensor: shape=(), dtype=float32, numpy=2.649808>}\n",
            "{'batch_loss': <tf.Tensor: shape=(), dtype=float32, numpy=2.5210183>}\n",
            "{'batch_loss': <tf.Tensor: shape=(), dtype=float32, numpy=2.3310845>}\n",
            "{'batch_loss': <tf.Tensor: shape=(), dtype=float32, numpy=2.0551789>}\n",
            "\n",
            "CPU times: user 38.8 s, sys: 730 ms, total: 39.5 s\n",
            "Wall time: 34.1 s\n"
          ]
        }
      ],
      "source": [
        "# Printing out the Batch loss of our model\n",
        "%%time\n",
        "for n in range(10):\n",
        "  print(translator.train_step([example_input_batch, example_target_batch]))\n",
        "print()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 300
        },
        "id": "KOl01ZMFnOsT",
        "outputId": "4d5c79bd-4404-4bd4-c257-cb95a596942b"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "....................................................................................................\n"
          ]
        },
        {
          "data": {
            "text/plain": [
              "[<matplotlib.lines.Line2D at 0x7fe068f7e250>]"
            ]
          },
          "execution_count": 67,
          "metadata": {},
          "output_type": "execute_result"
        },
        {
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "# Plotting our batch losses\n",
        "losses = []\n",
        "for n in range(100):\n",
        "  print('.', end='')\n",
        "  logs = translator.train_step([example_input_batch, example_target_batch])\n",
        "  losses.append(logs['batch_loss'].numpy())\n",
        "\n",
        "print()\n",
        "plt.plot(losses)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "LPHSFGQwnSCL"
      },
      "outputs": [],
      "source": [
        "# Building another model to train\n",
        "train_translator = TrainTranslator(\n",
        "    embedding_dim, units,\n",
        "    input_text_processor=input_text_processor,\n",
        "    output_text_processor=output_text_processor)\n",
        "\n",
        "# Configure the loss and optimizer\n",
        "train_translator.compile(\n",
        "    optimizer=tf.optimizers.Adam(),\n",
        "    loss=MaskedLoss(),\n",
        ")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "pY3YQsfan-O_"
      },
      "source": [
        "### iv) Train the model"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "TNhN4bJHoBSi"
      },
      "outputs": [],
      "source": [
        "# Training a couple of epochs by applying the callbacks.Callback method\n",
        "# to collect the history of batch losses\n",
        "class BatchLogs(tf.keras.callbacks.Callback):\n",
        "  def __init__(self, key):\n",
        "    self.key = key\n",
        "    self.logs = []\n",
        "\n",
        "  def on_train_batch_end(self, n, logs):\n",
        "    self.logs.append(logs[self.key])\n",
        "\n",
        "batch_loss = BatchLogs('batch_loss')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pgA08TIfoEvf",
        "outputId": "ea12e408-dacb-46eb-e248-95e7574800c3"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Epoch 1/15\n",
            "59/59 [==============================] - 124s 2s/step - batch_loss: 4.3935\n",
            "Epoch 2/15\n",
            "59/59 [==============================] - 117s 2s/step - batch_loss: 3.3654\n",
            "Epoch 3/15\n",
            "59/59 [==============================] - 117s 2s/step - batch_loss: 2.7471\n",
            "Epoch 4/15\n",
            "59/59 [==============================] - 118s 2s/step - batch_loss: 2.2088\n",
            "Epoch 5/15\n",
            "59/59 [==============================] - 117s 2s/step - batch_loss: 1.7998\n",
            "Epoch 6/15\n",
            "59/59 [==============================] - 117s 2s/step - batch_loss: 1.2770\n",
            "Epoch 7/15\n",
            "59/59 [==============================] - 117s 2s/step - batch_loss: 0.9423\n",
            "Epoch 8/15\n",
            "59/59 [==============================] - 117s 2s/step - batch_loss: 0.6300\n",
            "Epoch 9/15\n",
            "59/59 [==============================] - 118s 2s/step - batch_loss: 0.4450\n",
            "Epoch 10/15\n",
            "59/59 [==============================] - 116s 2s/step - batch_loss: 0.2977\n",
            "Epoch 11/15\n",
            "59/59 [==============================] - 117s 2s/step - batch_loss: 0.1798\n",
            "Epoch 12/15\n",
            "59/59 [==============================] - 116s 2s/step - batch_loss: 0.1055\n",
            "Epoch 13/15\n",
            "59/59 [==============================] - 116s 2s/step - batch_loss: 0.0571\n",
            "Epoch 14/15\n",
            "59/59 [==============================] - 117s 2s/step - batch_loss: 0.0328\n",
            "Epoch 15/15\n",
            "59/59 [==============================] - 116s 2s/step - batch_loss: 0.0221\n"
          ]
        },
        {
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7fe06a164e50>"
            ]
          },
          "execution_count": 70,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Displaying the batch loss using 15 epochs \n",
        "train_translator.fit(dataset, epochs=15,\n",
        "                     callbacks=[batch_loss])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 300
        },
        "id": "cPsGUSJkqWi6",
        "outputId": "093b9791-f357-49ad-e992-491daa9c6826"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "Text(0, 0.5, 'CE/token')"
            ]
          },
          "execution_count": 71,
          "metadata": {},
          "output_type": "execute_result"
        },
        {
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "# Plotting the epochs\n",
        "plt.plot(batch_loss.logs)\n",
        "plt.ylim([0, 3])\n",
        "plt.xlabel('Batch #')\n",
        "plt.ylabel('CE/token')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "079uq-8ZqcCF"
      },
      "source": [
        "##  Translate"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "pee0S4sxqbZL"
      },
      "outputs": [],
      "source": [
        "# Executing the full text => texttranslation\n",
        "# This is by inverting the text => token IDsmapping provided by the output_text_processor\n",
        "class Translator(tf.Module):\n",
        "\n",
        "  def __init__(self, encoder, decoder, input_text_processor,\n",
        "               output_text_processor):\n",
        "    self.encoder = encoder\n",
        "    self.decoder = decoder\n",
        "    self.input_text_processor = input_text_processor\n",
        "    self.output_text_processor = output_text_processor\n",
        "\n",
        "    self.output_token_string_from_index = (\n",
        "        tf.keras.layers.StringLookup(\n",
        "            vocabulary=output_text_processor.get_vocabulary(),\n",
        "            mask_token='',\n",
        "            invert=True))\n",
        "\n",
        "    # The output should never generate padding, unknown, or start.\n",
        "    index_from_string = tf.keras.layers.StringLookup(\n",
        "        vocabulary=output_text_processor.get_vocabulary(), mask_token='')\n",
        "    token_mask_ids = index_from_string(['', '[UNK]', '[START]']).numpy()\n",
        "\n",
        "    token_mask = np.zeros([index_from_string.vocabulary_size()], dtype=np.bool)\n",
        "    token_mask[np.array(token_mask_ids)] = True\n",
        "    self.token_mask = token_mask\n",
        "\n",
        "    self.start_token = index_from_string(tf.constant('[START]'))\n",
        "    self.end_token = index_from_string(tf.constant('[END]'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Q1zOGt-eqjgk",
        "outputId": "31ec3806-46dc-49db-c0b9-c9bb94bad1a8"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:23: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.\n",
            "Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations\n"
          ]
        }
      ],
      "source": [
        "translator = Translator(\n",
        "    encoder=train_translator.encoder,\n",
        "    decoder=train_translator.decoder,\n",
        "    input_text_processor=input_text_processor,\n",
        "    output_text_processor=output_text_processor,\n",
        ")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "pDZbxQB3qp3H"
      },
      "source": [
        "### i) Convert IDs to text"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "KphbGFO7q9Jc"
      },
      "outputs": [],
      "source": [
        "# Implementing the tokens_to_text which converts from token IDs to human readable text.\n",
        "def tokens_to_text(self, result_tokens):\n",
        "  shape_checker = ShapeChecker()\n",
        "  shape_checker(result_tokens, ('batch', 't'))\n",
        "  result_text_tokens = self.output_token_string_from_index(result_tokens)\n",
        "  shape_checker(result_text_tokens, ('batch', 't'))\n",
        "\n",
        "  result_text = tf.strings.reduce_join(result_text_tokens,\n",
        "                                       axis=1, separator=' ')\n",
        "  shape_checker(result_text, ('batch'))\n",
        "\n",
        "  result_text = tf.strings.strip(result_text)\n",
        "  shape_checker(result_text, ('batch',))\n",
        "  return result_text"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "NCmFMywfqsO7"
      },
      "outputs": [],
      "source": [
        "Translator.tokens_to_text = tokens_to_text"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Po_ThN89rEWF",
        "outputId": "606a7e7a-39a0-46f7-cdaf-5a0b0b106d8e"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "array([b'grieved above', b'oppress strengthen', b'shield enemies',\n",
              "       b'plead believed', b'directed statutes'], dtype=object)"
            ]
          },
          "execution_count": 76,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Inputting some random token IDs and see what it generates (example)\n",
        "example_output_tokens = tf.random.uniform(\n",
        "    shape=[5, 2], minval=0, dtype=tf.int64,\n",
        "    maxval=output_text_processor.vocabulary_size())\n",
        "translator.tokens_to_text(example_output_tokens).numpy()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "4muFfacgrL9E"
      },
      "source": [
        "### ii) Sample from the decoder's predictions"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "aYIHYKfJrKOz"
      },
      "outputs": [],
      "source": [
        "# Taking the decoder's logit outputs and samples token IDs from the distribution\n",
        "def sample(self, logits, temperature):\n",
        "  shape_checker = ShapeChecker()\n",
        "  # 't' is usually 1 here.\n",
        "  shape_checker(logits, ('batch', 't', 'vocab'))\n",
        "  shape_checker(self.token_mask, ('vocab',))\n",
        "\n",
        "  token_mask = self.token_mask[tf.newaxis, tf.newaxis, :]\n",
        "  shape_checker(token_mask, ('batch', 't', 'vocab'), broadcast=True)\n",
        "\n",
        "  # Set the logits for all masked tokens to -inf, so they are never chosen.\n",
        "  logits = tf.where(self.token_mask, -np.inf, logits)\n",
        "\n",
        "  if temperature == 0.0:\n",
        "    new_tokens = tf.argmax(logits, axis=-1)\n",
        "  else: \n",
        "    logits = tf.squeeze(logits, axis=1)\n",
        "    new_tokens = tf.random.categorical(logits/temperature,\n",
        "                                        num_samples=1)\n",
        "\n",
        "  shape_checker(new_tokens, ('batch', 't'))\n",
        "\n",
        "  return new_tokens"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "VteBiIP0rS8S"
      },
      "outputs": [],
      "source": [
        "Translator.sample = sample"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jNDKC8OyrZ2S",
        "outputId": "1b2ed842-81ea-4bda-94b9-e556d141522f"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "<tf.Tensor: shape=(5, 1), dtype=int64, numpy=\n",
              "array([[434],\n",
              "       [ 54],\n",
              "       [ 19],\n",
              "       [327],\n",
              "       [398]])>"
            ]
          },
          "execution_count": 79,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Random inputs (example)\n",
        "example_logits = tf.random.normal([5, 1, output_text_processor.vocabulary_size()])\n",
        "example_output_tokens = translator.sample(example_logits, temperature=1.0)\n",
        "example_output_tokens"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "j5Nr3PkorXYF"
      },
      "source": [
        "### iii) Implement translation loop"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "spIWGzjNrVsC"
      },
      "outputs": [],
      "source": [
        "# Taking the results into python lists before joining them  using tf.concat into tensors.\n",
        "# This unfolds the graph out to max_length iterations.\n",
        "def translate_unrolled(self,\n",
        "                       input_text, *,\n",
        "                       max_length=50,\n",
        "                       return_attention=True,\n",
        "                       temperature=1.0):\n",
        "  batch_size = tf.shape(input_text)[0]\n",
        "  input_tokens = self.input_text_processor(input_text)\n",
        "  enc_output, enc_state = self.encoder(input_tokens)\n",
        "\n",
        "  dec_state = enc_state\n",
        "  new_tokens = tf.fill([batch_size, 1], self.start_token)\n",
        "\n",
        "  result_tokens = []\n",
        "  attention = []\n",
        "  done = tf.zeros([batch_size, 1], dtype=tf.bool)\n",
        "\n",
        "  for _ in range(max_length):\n",
        "    dec_input = DecoderInput(new_tokens=new_tokens,\n",
        "                             enc_output=enc_output,\n",
        "                             mask=(input_tokens!=0))\n",
        "\n",
        "    dec_result, dec_state = self.decoder(dec_input, state=dec_state)\n",
        "\n",
        "    attention.append(dec_result.attention_weights)\n",
        "\n",
        "    new_tokens = self.sample(dec_result.logits, temperature)\n",
        "\n",
        "    # If a sequence produces an `end_token`, set it `done`\n",
        "    done = done | (new_tokens == self.end_token)\n",
        "    # Once a sequence is done it only produces 0-padding.\n",
        "    new_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), new_tokens)\n",
        "\n",
        "    # Collect the generated tokens\n",
        "    result_tokens.append(new_tokens)\n",
        "\n",
        "    if tf.executing_eagerly() and tf.reduce_all(done):\n",
        "      break\n",
        "\n",
        "  # Convert the list of generates token ids to a list of strings.\n",
        "  result_tokens = tf.concat(result_tokens, axis=-1)\n",
        "  result_text = self.tokens_to_text(result_tokens)\n",
        "\n",
        "  if return_attention:\n",
        "    attention_stack = tf.concat(attention, axis=1)\n",
        "    return {'text': result_text, 'attention': attention_stack}\n",
        "  else:\n",
        "    return {'text': result_text}"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "uSzYfqdFripi"
      },
      "outputs": [],
      "source": [
        "Translator.translate = translate_unrolled"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Gncwm1c6rlmS",
        "outputId": "e74255d0-2a25-47c0-b039-880b8c512b31"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "blessed from the undefiled in the way of the morning , and have taught me thy law .\n",
            "the proud have had for me , o lord fail are thy lovingkindness .\n",
            "\n",
            "CPU times: user 1.01 s, sys: 60.5 ms, total: 1.07 s\n",
            "Wall time: 864 ms\n"
          ]
        }
      ],
      "source": [
        "# Running a simple input to view the translation\n",
        "%%time\n",
        "input_text = tf.constant([\n",
        "    'Boiboen che igesunotgei eng’ oret.', # \"Blessed are the undefiled in the way.\"\n",
        "    'Kilosu Jehovah', # \"I have gone astray like a lost sheep\"\n",
        "])\n",
        "\n",
        "\n",
        "result = translator.translate(\n",
        "    input_text = input_text)\n",
        "\n",
        "print(result['text'][0].numpy().decode())\n",
        "print(result['text'][1].numpy().decode())\n",
        "print()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "uezLA_MYuRWg"
      },
      "source": [
        "##  Visualize the process"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5wP-I_t4uQQJ",
        "outputId": "b1c3e7a0-e3be-4516-bceb-02a39a56bb12"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "[1.         1.         1.         1.         0.99999994 1.\n",
            " 1.         1.         1.         1.         0.99999994 1.0000001\n",
            " 1.0000001  1.         1.0000001  0.99999994 1.0000001  1.\n",
            " 1.        ]\n"
          ]
        }
      ],
      "source": [
        "# Calculating the sum of the attention over the input which should return all ones.\n",
        "a = result['attention'][0]\n",
        "\n",
        "print(np.sum(a, axis=-1))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 265
        },
        "id": "bakU42xPuXL1",
        "outputId": "7ea2c723-e5f3-4a79-e197-eb69e7ff754d"
      },
      "outputs": [
        {
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "# The attention distribution for the first output step of the first example\n",
        "# It is focused than it was in the untrained model\n",
        "_ = plt.bar(range(len(a[0, :])), a[0, :])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        },
        "id": "GQVQ02YQubSE",
        "outputId": "d6f06ac6-adb6-4903-dba2-b03dd5aa0d82"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "<matplotlib.image.AxesImage at 0x7fe069b73c50>"
            ]
          },
          "execution_count": 85,
          "metadata": {},
          "output_type": "execute_result"
        },
        {
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "# There is some rough alignment between the input and output words\n",
        "plt.imshow(np.array(a), vmin=0.0)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "X4Ff_LgNui5O"
      },
      "source": [
        "### i) Labelled attention plots"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "JLFvBR__u5hv"
      },
      "outputs": [],
      "source": [
        "# Visualizing the attention plots.\n",
        "def plot_attention(attention, sentence, predicted_sentence):\n",
        "  sentence = tf_lower_and_split_punct(sentence).numpy().decode().split()\n",
        "  predicted_sentence = predicted_sentence.numpy().decode().split() + ['[END]']\n",
        "  fig = plt.figure(figsize=(10, 10))\n",
        "  ax = fig.add_subplot(1, 1, 1)\n",
        "\n",
        "  attention = attention[:len(predicted_sentence), :len(sentence)]\n",
        "\n",
        "  ax.matshow(attention, cmap='viridis', vmin=0.0)\n",
        "\n",
        "  fontdict = {'fontsize': 14}\n",
        "\n",
        "  ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)\n",
        "  ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)\n",
        "\n",
        "  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))\n",
        "  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))\n",
        "\n",
        "  ax.set_xlabel('Input text')\n",
        "  ax.set_ylabel('Output text')\n",
        "  plt.suptitle('Attention weights')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 678
        },
        "id": "xtk8wPMduev-",
        "outputId": "d4ebbc89-6d21-49f0-f37d-99ffce48f5b3"
      },
      "outputs": [
        {
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 720x720 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "i=0\n",
        "plot_attention(result['attention'][i], input_text[i], result['text'][i])"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Zlq2VEE3p-fi"
      },
      "source": [
        "## Export"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "a2cHjk7tp9gT"
      },
      "outputs": [],
      "source": [
        "@tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])\n",
        "def tf_translate(self, input_text):\n",
        "  return self.translate(input_text)\n",
        "\n",
        "Translator.tf_translate = tf_translate"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0tcB73funwpf"
      },
      "source": [
        "##  Conclusion"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Rj2G7roFn1hN"
      },
      "source": [
        "a). Did we have the right data?\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "MjuvLmo6bF01"
      },
      "source": [
        "b). Do we need other data to answer our question?"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "rS0AjWVvbaPM"
      },
      "source": [
        "c) Did we have the right question?"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tL_GNDpjYTQu"
      },
      "source": [
        "## Installations"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "5VMXz4ltYRWv"
      },
      "outputs": [],
      "source": [
        "# pip install ipykernel>=5.1.2\n",
        "# pip install pydeck\n",
        "# pip install streamlit==0.75.0\n",
        "# pip install pyngrok\n",
        "# pip install streamlit -q\n",
        "# pip install streamlit --upgrade\n",
        "# pip install streamlit-option-menu\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6k7ASCARZHGj"
      },
      "source": [
        "## App"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "l9hEQcRVYPIu",
        "outputId": "901ebc2b-e33c-4b9e-c239-49563a314133"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Overwriting finalLHTranslation.py\n"
          ]
        }
      ],
      "source": [
        "%%writefile finalLHTranslation.py\n",
        "py -m pip install tensorflow\n",
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import plotly.figure_factory as ff\n",
        "from streamlit_option_menu import option_menu\n",
        "from textblob import TextBlob \n",
        "import tensorflow as tf\n",
        " \n",
        " \n",
        "\n",
        "# let's do the navigation bar first\n",
        "\n",
        "selected = option_menu(\n",
        "      menu_title= None, options=['Home','Features', 'About'], icons =['house','book','boxes'],menu_icon='cast', default_index=0, orientation = 'horizontal'\n",
        "  )\n",
        "\n",
        "\n",
        "#theme\n",
        "CURRENT_THEME = \"light\"\n",
        "IS_DARK_THEME = False\n",
        "EXPANDER_TEXT = \"\"\"\n",
        "    This is Streamlit's default *Light* theme. It should be enabled by default 🎈\n",
        "    If not, you can enable it in the app menu (☰ -> Settings -> Theme).\n",
        "    \"\"\"\n",
        "\n",
        "# setting containers\n",
        "header = st.container()\n",
        "translation = st.container()\n",
        "dataset = st.container()\n",
        "features = st.container()\n",
        "modelTraining = st.container()\n",
        "\n",
        "with header:\n",
        "  col1, col2 = st.columns([1,6])\n",
        "\n",
        "  with col1:\n",
        "    st.image(\n",
        "    \"https://cdn0.iconfinder.com/data/icons/joker-circus-by-joker2011-d3g8h6s/256/lion.png\", width=100,)\n",
        "\n",
        "  with col2:\n",
        "    st.markdown(\"<h1 style='text-align: left; color: Orange;'> Lion Heart Translation</h1>\", unsafe_allow_html=True)\n",
        "\n",
        "\n",
        "\n",
        "  # st.image(\n",
        "  #   \"https://cdn0.iconfinder.com/data/icons/joker-circus-by-joker2011-d3g8h6s/256/lion.png\", width=100,)\n",
        "  # st.markdown(\"<h1 style='text-align: center; color: Purple;'> Lion Heart Translation</h1>\", unsafe_allow_html=True)\n",
        "  # st.text(\"Lion Heart Translation is an App who's main porpose is to translate Kenyan local languages, that is Kalenjin, Luo and Kikuyu to English and vise versa. It's Using a tensorflow neural network model in order to do this.\")\n",
        "with st.expander(\"Pick out a theme of your liking?\"):\n",
        "  THEMES = [\n",
        "    \"light\",\n",
        "    \"dark\",\n",
        "    \"green\",\n",
        "    \"blue\",]\n",
        "  GITHUB_OWNER = \"streamlit\"\n",
        "\n",
        "\n",
        "\n",
        "# modeling ie translation code\n",
        "if selected == 'Home':\n",
        "  with translation:\n",
        "    from textblob import TextBlob \n",
        "    import spacy\n",
        "    from gensim.summarization import summarize\n",
        "    sp = spacy.load('en_core_web_sm')\n",
        "    from spacy import displacy\n",
        " \n",
        "    # Add selectbox in streamlit\n",
        "    st.markdown(\"\"\"<span style=\"word-wrap:break-word;\">Lion Heart Translation is an App who's main porpose is to translate Kenyan local languages, that is Kalenjin, Luo and Kikuyu to English and vise versa. It's Using a tensorflow neural network model in order to do this.</span>\"\"\", unsafe_allow_html=True)\n",
        "    option = st.selectbox(\n",
        "     'Which Local language would you like to translate to?',\n",
        "        ('none', 'Kikuyu', 'Kalenjin', 'Luo'))\n",
        "    st.write('You selected:', option)\n",
        "\n",
        "    def main():\n",
        "      text = st.text_area(\"Enter Text to translate here: \",\"lorem ipsum...\",key = \"<255>\")\n",
        "      if st.button(\"Translate\"):\n",
        "        input_text = tf.constant(text)\n",
        "        result = translator.translate(input_text = input_text)\n",
        "        show = result['text'][0].numpy().decode()\n",
        "        st.success(show)\n",
        "\n",
        "if __name__=='__main__':\n",
        "  main()\n",
        "\n",
        "\n",
        "if selected == 'About':\n",
        "  with dataset:\n",
        "    st.header('About')\n",
        "    st.markdown(\"\"\"<span style=\"word-wrap:break-word;\">The data used to build this model was obtained from a chapter in the Bible in each of the four languages.</span>\"\"\", unsafe_allow_html=True)\n",
        "    \n",
        "    # st.text('The data used to build this model was obtained from a chapter in the Bible in each of the four languages.')\n",
        "    st.text(\"here's what the Kalenjin looks side by side with it's English translation\")\n",
        "    kaleme = pd.read_csv('/content/kaleme.csv', sep='delimiter', engine = 'python', header=None)\n",
        "    st.write(kaleme.head(5))\n",
        "\n",
        "if selected == 'Features':\n",
        "  with features:\n",
        "    st.header('Features')\n",
        "    st.markdown('* **Hyperparameter tuning:** Here the user can tweak the model settings in pursuit of higher accuracy')\n",
        "    st.markdown('* **Language Dropdown:** Here the user can tweak the model settings in pursuit of higher accuracy')\n",
        "    st.markdown('* **Translation textbox:** Here the user can tweak the model settings in pursuit of higher accuracy')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true,
          "base_uri": "https://localhost:8080/"
        },
        "id": "UNJf5P3BZDYd",
        "outputId": "76761afb-9a77-46d1-c65b-05fd524d9f40"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "2022-05-19 05:15:21.860 INFO    numexpr.utils: NumExpr defaulting to 2 threads.\n",
            "\u001b[K\u001b[?25hnpx: installed 22 in 2.63s\n",
            "\u001b[0m\n",
            "\u001b[34m\u001b[1m  You can now view your Streamlit app in your browser.\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Network URL: \u001b[0m\u001b[1mhttp://172.28.0.2:8501\u001b[0m\n",
            "\u001b[34m  External URL: \u001b[0m\u001b[1mhttp://34.83.84.125:8501\u001b[0m\n",
            "\u001b[0m\n",
            "your url is: https://icy-teeth-clean-34-83-84-125.loca.lt\n",
            "2022-05-19 06:17:38.750 'pattern' package not found; tag filters are not available for English\n",
            "2022-05-19 06:19:30.374302: E tensorflow/stream_executor/cuda/cuda_driver.cc:271] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected\n",
            "2022-05-19 06:19:30.377 Traceback (most recent call last):\n",
            "  File \"/usr/local/lib/python3.7/dist-packages/streamlit/scriptrunner/script_runner.py\", line 475, in _run_script\n",
            "    exec(code, module.__dict__)\n",
            "  File \"/content/finalLHTranslation.py\", line 83, in <module>\n",
            "    main()\n",
            "  File \"/content/finalLHTranslation.py\", line 78, in main\n",
            "    result = translator.translate(input_text = input_text)\n",
            "NameError: name 'translator' is not defined\n",
            "\n"
          ]
        }
      ],
      "source": [
        "# ! streamlit run finalLHTranslation.py & npx localtunnel --port 8501\n"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "collapsed_sections": [],
      "name": "Final App Translation",
      "provenance": [],
      "toc_visible": true,
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}