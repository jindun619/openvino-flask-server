{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "8402a84a-e41a-4767-a4aa-498bffdcefe5",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple\n",
      "Collecting ipywidgets\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/53/b8/62952729573d983d9433faacf62a52ee2e8cf46504418061ad1739967abe/ipywidgets-8.1.6-py3-none-any.whl (139 kB)\n",
      "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m139.8/139.8 kB\u001b[0m \u001b[31m16.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
      "\u001b[?25hRequirement already satisfied: comm>=0.1.3 in /environment/miniconda3/lib/python3.11/site-packages (from ipywidgets) (0.2.2)\n",
      "Requirement already satisfied: ipython>=6.1.0 in /environment/miniconda3/lib/python3.11/site-packages (from ipywidgets) (8.23.0)\n",
      "Requirement already satisfied: traitlets>=4.3.1 in /environment/miniconda3/lib/python3.11/site-packages (from ipywidgets) (5.14.3)\n",
      "Collecting widgetsnbextension~=4.0.14 (from ipywidgets)\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ca/51/5447876806d1088a0f8f71e16542bf350918128d0a69437df26047c8e46f/widgetsnbextension-4.0.14-py3-none-any.whl (2.2 MB)\n",
      "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m2.2/2.2 MB\u001b[0m \u001b[31m85.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
      "\u001b[?25hCollecting jupyterlab_widgets~=3.0.14 (from ipywidgets)\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/64/7a/f2479ba401e02f7fcbd3fc6af201eac888eaa188574b8e9df19452ab4972/jupyterlab_widgets-3.0.14-py3-none-any.whl (213 kB)\n",
      "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m214.0/214.0 kB\u001b[0m \u001b[31m25.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
      "\u001b[?25hRequirement already satisfied: decorator in /environment/miniconda3/lib/python3.11/site-packages (from ipython>=6.1.0->ipywidgets) (5.1.1)\n",
      "Requirement already satisfied: jedi>=0.16 in /environment/miniconda3/lib/python3.11/site-packages (from ipython>=6.1.0->ipywidgets) (0.19.1)\n",
      "Requirement already satisfied: matplotlib-inline in /environment/miniconda3/lib/python3.11/site-packages (from ipython>=6.1.0->ipywidgets) (0.1.7)\n",
      "Requirement already satisfied: prompt-toolkit<3.1.0,>=3.0.41 in /environment/miniconda3/lib/python3.11/site-packages (from ipython>=6.1.0->ipywidgets) (3.0.43)\n",
      "Requirement already satisfied: pygments>=2.4.0 in /environment/miniconda3/lib/python3.11/site-packages (from ipython>=6.1.0->ipywidgets) (2.17.2)\n",
      "Requirement already satisfied: stack-data in /environment/miniconda3/lib/python3.11/site-packages (from ipython>=6.1.0->ipywidgets) (0.6.3)\n",
      "Requirement already satisfied: typing-extensions in /environment/miniconda3/lib/python3.11/site-packages (from ipython>=6.1.0->ipywidgets) (4.13.2)\n",
      "Requirement already satisfied: pexpect>4.3 in /environment/miniconda3/lib/python3.11/site-packages (from ipython>=6.1.0->ipywidgets) (4.9.0)\n",
      "Requirement already satisfied: parso<0.9.0,>=0.8.3 in /environment/miniconda3/lib/python3.11/site-packages (from jedi>=0.16->ipython>=6.1.0->ipywidgets) (0.8.4)\n",
      "Requirement already satisfied: ptyprocess>=0.5 in /environment/miniconda3/lib/python3.11/site-packages (from pexpect>4.3->ipython>=6.1.0->ipywidgets) (0.7.0)\n",
      "Requirement already satisfied: wcwidth in /environment/miniconda3/lib/python3.11/site-packages (from prompt-toolkit<3.1.0,>=3.0.41->ipython>=6.1.0->ipywidgets) (0.2.13)\n",
      "Requirement already satisfied: executing>=1.2.0 in /environment/miniconda3/lib/python3.11/site-packages (from stack-data->ipython>=6.1.0->ipywidgets) (2.0.1)\n",
      "Requirement already satisfied: asttokens>=2.1.0 in /environment/miniconda3/lib/python3.11/site-packages (from stack-data->ipython>=6.1.0->ipywidgets) (2.4.1)\n",
      "Requirement already satisfied: pure-eval in /environment/miniconda3/lib/python3.11/site-packages (from stack-data->ipython>=6.1.0->ipywidgets) (0.2.2)\n",
      "Requirement already satisfied: six>=1.12.0 in /environment/miniconda3/lib/python3.11/site-packages (from asttokens>=2.1.0->stack-data->ipython>=6.1.0->ipywidgets) (1.16.0)\n",
      "Installing collected packages: widgetsnbextension, jupyterlab_widgets, ipywidgets\n",
      "Successfully installed ipywidgets-8.1.6 jupyterlab_widgets-3.0.14 widgetsnbextension-4.0.14\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[33m(Deprecated) Installing extensions with the jupyter labextension install command is now deprecated and will be removed in a future major version of JupyterLab.\n",
      "\n",
      "Users should manage prebuilt extensions with package managers like pip and conda, and extension authors are encouraged to distribute their extensions as prebuilt packages \u001b[0m\n",
      "Building jupyterlab assets (production, minimized)\n",
      "/environment/miniconda3/lib/python3.11/site-packages/jupyterlab/debuglog.py:54: UserWarning: An error occurred.\n",
      "  warnings.warn(\"An error occurred.\")\n",
      "/environment/miniconda3/lib/python3.11/site-packages/jupyterlab/debuglog.py:55: UserWarning: RuntimeError: npm dependencies failed to install\n",
      "  warnings.warn(msg[-1].strip())\n",
      "/environment/miniconda3/lib/python3.11/site-packages/jupyterlab/debuglog.py:56: UserWarning: See the log file for details: /tmp/jupyterlab-debug-sbikmkeu.log\n",
      "  warnings.warn(f\"See the log file for details: {log_path!s}\")\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "76ff13c4f7774afba5c9bd5925c92c5d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Add pad token = ['<｜▁pad▁｜>'] to the tokenizer\n",
      "<｜▁pad▁｜>:2\n",
      "Add image token = ['<image>'] to the tokenizer\n",
      "<image>:128815\n",
      "Add grounding-related tokens = ['<|ref|>', '<|/ref|>', '<|det|>', '<|/det|>', '<|grounding|>'] to the tokenizer with input_ids\n",
      "<|ref|>:128816\n",
      "<|/ref|>:128817\n",
      "<|det|>:128818\n",
      "<|/det|>:128819\n",
      "<|grounding|>:128820\n",
      "Add chat tokens = ['<|User|>', '<|Assistant|>'] to the tokenizer with input_ids\n",
      "<|User|>:128821\n",
      "<|Assistant|>:128822\n",
      "\n"
     ]
    }
   ],
   "source": [
    "!pip install ipywidgets\n",
    "!jupyter labextension install @jupyter-widgets/jupyterlab-manager\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "077038be-7e3c-4dcd-86cd-d38964cd6b06",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple\n",
      "Requirement already satisfied: openvino>=2025.0 in /environment/miniconda3/lib/python3.11/site-packages (2025.1.0)\n",
      "Collecting nncf>=2.15\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/03/1a/9a2dd53b1661d3deb191e607fdad689687515761475ec9f1770eaafbdec3/nncf-2.16.0-py3-none-any.whl (1.4 MB)\n",
      "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m1.4/1.4 MB\u001b[0m \u001b[31m51.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m:00:01\u001b[0m\n",
      "\u001b[?25hRequirement already satisfied: numpy<2.3.0,>=1.16.6 in /environment/miniconda3/lib/python3.11/site-packages (from openvino>=2025.0) (1.26.4)\n",
      "Requirement already satisfied: openvino-telemetry>=2023.2.1 in /environment/miniconda3/lib/python3.11/site-packages (from openvino>=2025.0) (2025.1.0)\n",
      "Requirement already satisfied: packaging in /environment/miniconda3/lib/python3.11/site-packages (from openvino>=2025.0) (23.2)\n",
      "Requirement already satisfied: jsonschema>=3.2.0 in /environment/miniconda3/lib/python3.11/site-packages (from nncf>=2.15) (4.21.1)\n",
      "Collecting jstyleson>=0.0.2 (from nncf>=2.15)\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/5f/c1/3791e39d65c39b4e081010f2cf6f9443b7b6b9f6de63a5e458172c5c9376/jstyleson-0.0.2.tar.gz (2.0 kB)\n",
      "  Preparing metadata (setup.py) ... \u001b[?25ldone\n",
      "\u001b[?25hCollecting natsort>=7.1.0 (from nncf>=2.15)\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ef/82/7a9d0550484a62c6da82858ee9419f3dd1ccc9aa1c26a1e43da3ecd20b0d/natsort-8.4.0-py3-none-any.whl (38 kB)\n",
      "Requirement already satisfied: networkx<3.5.0,>=2.6 in /environment/miniconda3/lib/python3.11/site-packages (from nncf>=2.15) (3.3)\n",
      "Collecting ninja<1.12,>=1.10.0.post2 (from nncf>=2.15)\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/eb/7a/455d2877fe6cf99886849c7f9755d897df32eaf3a0fba47b56e615f880f7/ninja-1.11.1.4-py3-none-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (422 kB)\n",
      "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m422.8/422.8 kB\u001b[0m \u001b[31m34.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
      "\u001b[?25hRequirement already satisfied: pandas<2.3,>=1.1.5 in /environment/miniconda3/lib/python3.11/site-packages (from nncf>=2.15) (2.2.2)\n",
      "Requirement already satisfied: psutil in /environment/miniconda3/lib/python3.11/site-packages (from nncf>=2.15) (5.9.8)\n",
      "Collecting pydot<=3.0.4,>=1.4.1 (from nncf>=2.15)\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b0/5f/1ebfd430df05c4f9e438dd3313c4456eab937d976f6ab8ce81a98f9fb381/pydot-3.0.4-py3-none-any.whl (35 kB)\n",
      "Collecting pymoo>=0.6.0.1 (from nncf>=2.15)\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/43/b8/b709e68f663ef1aaf0187cfcb688de1c31e182a4f55ce90064acec451726/pymoo-0.6.1.3-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.4 MB)\n",
      "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m4.4/4.4 MB\u001b[0m \u001b[31m53.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m00:01\u001b[0m00:01\u001b[0mm\n",
      "\u001b[?25hRequirement already satisfied: rich>=13.5.2 in /environment/miniconda3/lib/python3.11/site-packages (from nncf>=2.15) (13.7.1)\n",
      "Requirement already satisfied: safetensors>=0.4.1 in ./work/.local/lib/python3.11/site-packages (from nncf>=2.15) (0.5.3)\n",
      "Collecting scikit-learn>=0.24.0 (from nncf>=2.15)\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a8/f3/62fc9a5a659bb58a03cdd7e258956a5824bdc9b4bb3c5d932f55880be569/scikit_learn-1.6.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (13.5 MB)\n",
      "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m13.5/13.5 MB\u001b[0m \u001b[31m80.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m00:01\u001b[0m00:01\u001b[0m\n",
      "\u001b[?25hCollecting scipy>=1.3.2 (from nncf>=2.15)\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/32/ea/564bacc26b676c06a00266a3f25fdfe91a9d9a2532ccea7ce6dd394541bc/scipy-1.15.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (37.6 MB)\n",
      "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m37.6/37.6 MB\u001b[0m \u001b[31m54.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m00:01\u001b[0m00:01\u001b[0m\n",
      "\u001b[?25hCollecting tabulate>=0.9.0 (from nncf>=2.15)\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/40/44/4a5f08c96eb108af5cb50b41f76142f0afa346dfa99d5296fe7202a11854/tabulate-0.9.0-py3-none-any.whl (35 kB)\n",
      "Requirement already satisfied: attrs>=22.2.0 in /environment/miniconda3/lib/python3.11/site-packages (from jsonschema>=3.2.0->nncf>=2.15) (23.2.0)\n",
      "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /environment/miniconda3/lib/python3.11/site-packages (from jsonschema>=3.2.0->nncf>=2.15) (2023.12.1)\n",
      "Requirement already satisfied: referencing>=0.28.4 in /environment/miniconda3/lib/python3.11/site-packages (from jsonschema>=3.2.0->nncf>=2.15) (0.35.0)\n",
      "Requirement already satisfied: rpds-py>=0.7.1 in /environment/miniconda3/lib/python3.11/site-packages (from jsonschema>=3.2.0->nncf>=2.15) (0.18.0)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in /environment/miniconda3/lib/python3.11/site-packages (from pandas<2.3,>=1.1.5->nncf>=2.15) (2.9.0.post0)\n",
      "Requirement already satisfied: pytz>=2020.1 in /environment/miniconda3/lib/python3.11/site-packages (from pandas<2.3,>=1.1.5->nncf>=2.15) (2024.1)\n",
      "Requirement already satisfied: tzdata>=2022.7 in /environment/miniconda3/lib/python3.11/site-packages (from pandas<2.3,>=1.1.5->nncf>=2.15) (2024.1)\n",
      "Requirement already satisfied: pyparsing>=3.0.9 in /environment/miniconda3/lib/python3.11/site-packages (from pydot<=3.0.4,>=1.4.1->nncf>=2.15) (3.1.2)\n",
      "Requirement already satisfied: matplotlib>=3 in /environment/miniconda3/lib/python3.11/site-packages (from pymoo>=0.6.0.1->nncf>=2.15) (3.9.0)\n",
      "Collecting autograd>=1.4 (from pymoo>=0.6.0.1->nncf>=2.15)\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/6d/90/d13cf396989052cadd8511c1878b0913bbce28eeef5feb95710a92e03076/autograd-1.7.0-py3-none-any.whl (52 kB)\n",
      "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m52.5/52.5 kB\u001b[0m \u001b[31m7.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
      "\u001b[?25hCollecting cma==3.2.2 (from pymoo>=0.6.0.1->nncf>=2.15)\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/3d/d9/e607fd2257cc18d64ed4634d3fc437fda81cbd6d728f2d7aa5aad3a95447/cma-3.2.2-py2.py3-none-any.whl (249 kB)\n",
      "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m249.1/249.1 kB\u001b[0m \u001b[31m154.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
      "\u001b[?25hCollecting alive-progress (from pymoo>=0.6.0.1->nncf>=2.15)\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/57/39/cade3a5a97fffa3ae84f298208237b3a9f7112d6b0ed57e8ff4b755e44b4/alive_progress-3.2.0-py3-none-any.whl (77 kB)\n",
      "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m77.1/77.1 kB\u001b[0m \u001b[31m9.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
      "\u001b[?25hCollecting dill (from pymoo>=0.6.0.1->nncf>=2.15)\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/50/3d/9373ad9c56321fdab5b41197068e1d8c25883b3fea29dd361f9b55116869/dill-0.4.0-py3-none-any.whl (119 kB)\n",
      "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m119.7/119.7 kB\u001b[0m \u001b[31m17.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
      "\u001b[?25hCollecting Deprecated (from pymoo>=0.6.0.1->nncf>=2.15)\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/6e/c6/ac0b6c1e2d138f1002bcf799d330bd6d85084fece321e662a14223794041/Deprecated-1.2.18-py2.py3-none-any.whl (10.0 kB)\n",
      "Requirement already satisfied: markdown-it-py>=2.2.0 in /environment/miniconda3/lib/python3.11/site-packages (from rich>=13.5.2->nncf>=2.15) (3.0.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /environment/miniconda3/lib/python3.11/site-packages (from rich>=13.5.2->nncf>=2.15) (2.17.2)\n",
      "Collecting joblib>=1.2.0 (from scikit-learn>=0.24.0->nncf>=2.15)\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/91/29/df4b9b42f2be0b623cbd5e2140cafcaa2bef0759a00b7b70104dcfe2fb51/joblib-1.4.2-py3-none-any.whl (301 kB)\n",
      "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m301.8/301.8 kB\u001b[0m \u001b[31m29.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
      "\u001b[?25hCollecting threadpoolctl>=3.1.0 (from scikit-learn>=0.24.0->nncf>=2.15)\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/32/d5/f9a850d79b0851d1d4ef6456097579a9005b31fea68726a4ae5f2d82ddd9/threadpoolctl-3.6.0-py3-none-any.whl (18 kB)\n",
      "Requirement already satisfied: mdurl~=0.1 in /environment/miniconda3/lib/python3.11/site-packages (from markdown-it-py>=2.2.0->rich>=13.5.2->nncf>=2.15) (0.1.2)\n",
      "Requirement already satisfied: contourpy>=1.0.1 in /environment/miniconda3/lib/python3.11/site-packages (from matplotlib>=3->pymoo>=0.6.0.1->nncf>=2.15) (1.2.1)\n",
      "Requirement already satisfied: cycler>=0.10 in /environment/miniconda3/lib/python3.11/site-packages (from matplotlib>=3->pymoo>=0.6.0.1->nncf>=2.15) (0.12.1)\n",
      "Requirement already satisfied: fonttools>=4.22.0 in /environment/miniconda3/lib/python3.11/site-packages (from matplotlib>=3->pymoo>=0.6.0.1->nncf>=2.15) (4.53.0)\n",
      "Requirement already satisfied: kiwisolver>=1.3.1 in /environment/miniconda3/lib/python3.11/site-packages (from matplotlib>=3->pymoo>=0.6.0.1->nncf>=2.15) (1.4.5)\n",
      "Requirement already satisfied: pillow>=8 in /environment/miniconda3/lib/python3.11/site-packages (from matplotlib>=3->pymoo>=0.6.0.1->nncf>=2.15) (10.3.0)\n",
      "Requirement already satisfied: six>=1.5 in /environment/miniconda3/lib/python3.11/site-packages (from python-dateutil>=2.8.2->pandas<2.3,>=1.1.5->nncf>=2.15) (1.16.0)\n",
      "Collecting about-time==4.2.1 (from alive-progress->pymoo>=0.6.0.1->nncf>=2.15)\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/fb/cd/7ee00d6aa023b1d0551da0da5fee3bc23c3eeea632fbfc5126d1fec52b7e/about_time-4.2.1-py3-none-any.whl (13 kB)\n",
      "Collecting grapheme==0.6.0 (from alive-progress->pymoo>=0.6.0.1->nncf>=2.15)\n",
      "  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ce/e7/bbaab0d2a33e07c8278910c1d0d8d4f3781293dfbc70b5c38197159046bf/grapheme-0.6.0.tar.gz (207 kB)\n",
      "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m207.3/207.3 kB\u001b[0m \u001b[31m162.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
      "\u001b[?25h  Preparing metadata (setup.py) ... \u001b[?25ldone\n",
      "\u001b[?25hRequirement already satisfied: wrapt<2,>=1.10 in /environment/miniconda3/lib/python3.11/site-packages (from Deprecated->pymoo>=0.6.0.1->nncf>=2.15) (1.16.0)\n",
      "Building wheels for collected packages: jstyleson, grapheme\n",
      "  Building wheel for jstyleson (setup.py) ... \u001b[?25ldone\n",
      "\u001b[?25h  Created wheel for jstyleson: filename=jstyleson-0.0.2-py3-none-any.whl size=2384 sha256=b75b9df97640513c32e2b615e27193ab179dee3267d788074353dd98537e9412\n",
      "  Stored in directory: /home/featurize/.cache/pip/wheels/3b/34/60/d06a2a7e336927b1c354fff1f76d1e2e0c7c30420d909a1b99\n",
      "  Building wheel for grapheme (setup.py) ... \u001b[?25ldone\n",
      "\u001b[?25h  Created wheel for grapheme: filename=grapheme-0.6.0-py3-none-any.whl size=210079 sha256=d6774d26fd54485f3ffe3893ec46913e2bbde633ced676739d3ac8115b0b7fc3\n",
      "  Stored in directory: /home/featurize/.cache/pip/wheels/64/5c/5d/98c3a7ed8621c72d4050d91f915ef79a75bc047afe230d1ecd\n",
      "Successfully built jstyleson grapheme\n",
      "Installing collected packages: jstyleson, grapheme, threadpoolctl, tabulate, scipy, pydot, ninja, natsort, joblib, dill, Deprecated, cma, autograd, about-time, scikit-learn, alive-progress, pymoo, nncf\n",
      "Successfully installed Deprecated-1.2.18 about-time-4.2.1 alive-progress-3.2.0 autograd-1.7.0 cma-3.2.2 dill-0.4.0 grapheme-0.6.0 joblib-1.4.2 jstyleson-0.0.2 natsort-8.4.0 ninja-1.11.1.4 nncf-2.16.0 pydot-3.0.4 pymoo-0.6.1.3 scikit-learn-1.6.1 scipy-1.15.2 tabulate-0.9.0 threadpoolctl-3.6.0\n"
     ]
    }
   ],
   "source": [
    "!pip install -U \"openvino>=2025.0\" \"nncf>=2.15\"\n",
    "!pip install -q \\\n",
    "    \"torch>=2.1\" \\\n",
    "    \"torchvision\" \\\n",
    "    \"gradio>=4.19\" \\\n",
    "    \"einops\" \\\n",
    "    \"transformers>=4.48.2\" \\\n",
    "    \"timm>=0.9.16\" \\\n",
    "    \"accelerate\" \\\n",
    "    \"sentencepiece\" \\\n",
    "    \"attrdict\" \\\n",
    "    \"mdtex2html\" \\\n",
    "    \"pypinyin\" \\\n",
    "    \"tiktoken\" \\\n",
    "    \"tqdm\" \\\n",
    "    \"colorama\" \\\n",
    "    \"Pygments\" \\\n",
    "    \"markdown\" \\\n",
    "    --extra-index-url https://download.pytorch.org/whl/cpu\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "141dab16-2228-4188-961e-f8ec74096a21",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1단계. 유틸리티 및 헬퍼 파일 다운로드"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "073e127e-b4a1-4a50-9c85-3df5d726ead1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 필요한 helper 파일 자동 다운로드\n",
    "import requests\n",
    "from pathlib import Path\n",
    "\n",
    "# 공통 유틸\n",
    "utility_files = [\"cmd_helper.py\", \"notebook_utils.py\", \"pip_helper.py\"]\n",
    "base_utility_url = \"https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/\"\n",
    "\n",
    "for utility in utility_files:\n",
    "    if not Path(utility).exists():\n",
    "        r = requests.get(base_utility_url + utility)\n",
    "        with open(utility, \"w\", encoding=\"utf-8\") as f:\n",
    "            f.write(r.text)\n",
    "\n",
    "# DeepSeek-VL2 전용 헬퍼\n",
    "helper_files = [\"ov_deepseek_vl_helper.py\", \"modeling_helper.py\", \"gradio_helper.py\"]\n",
    "base_helper_url = \"https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/deepseek-vl2/\"\n",
    "\n",
    "for helper in helper_files:\n",
    "    if not Path(helper).exists():\n",
    "        r = requests.get(base_helper_url + helper)\n",
    "        with open(helper, \"w\", encoding=\"utf-8\") as f:\n",
    "            f.write(r.text)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "e89f9498-f807-48f7-8234-71c4f7670bef",
   "metadata": {},
   "outputs": [],
   "source": [
    "#2단계. 모델 준비 및 변환"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "a77ed119-6a31-4af6-ac08-47c7b4d48a8b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/environment/miniconda3/lib/python3.11/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html\n",
      "  from .autonotebook import tqdm as notebook_tqdm\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Python version is above 3.10, patching the collections module.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-04-16 06:21:43.397452: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered\n",
      "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR\n",
      "E0000 00:00:1744784503.414349    3435 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered\n",
      "E0000 00:00:1744784503.419644    3435 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered\n",
      "W0000 00:00:1744784503.432059    3435 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.\n",
      "W0000 00:00:1744784503.432072    3435 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.\n",
      "W0000 00:00:1744784503.432074    3435 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.\n",
      "W0000 00:00:1744784503.432076    3435 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.\n",
      "2025-04-16 06:21:43.436407: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.\n",
      "To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "⌛ deepseek-ai/deepseek-vl2-tiny conversion started. Be patient, it may takes some time.\n",
      "⌛ Load Original model\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "DeepseekVLV2ForCausalLM has generative capabilities, as `prepare_inputs_for_generation` is explicitly overwritten. However, it doesn't directly inherit from `GenerationMixin`. From 👉v4.50👈 onwards, `PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability to call `generate` and other related functions.\n",
      "  - If you're using `trust_remote_code=True`, you can get rid of this warning by loading the model with an auto class. See https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes\n",
      "  - If you are the owner of the model architecture code, please modify your model class such that it inherits from `GenerationMixin` (after `PreTrainedModel`, otherwise you'll get an exception).\n",
      "  - If you are not the owner of the model architecture class, please contact the model code owner to update it.\n",
      "DeepseekV2ForCausalLM has generative capabilities, as `prepare_inputs_for_generation` is explicitly overwritten. However, it doesn't directly inherit from `GenerationMixin`. From 👉v4.50👈 onwards, `PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability to call `generate` and other related functions.\n",
      "  - If you're using `trust_remote_code=True`, you can get rid of this warning by loading the model with an auto class. See https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes\n",
      "  - If you are the owner of the model architecture code, please modify your model class such that it inherits from `GenerationMixin` (after `PreTrainedModel`, otherwise you'll get an exception).\n",
      "  - If you are not the owner of the model architecture class, please contact the model code owner to update it.\n",
      "You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565 - if you loaded a llama tokenizer from a GGUF file you can ignore this message.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Add pad token = ['<｜▁pad▁｜>'] to the tokenizer\n",
      "<｜▁pad▁｜>:2\n",
      "Add image token = ['<image>'] to the tokenizer\n",
      "<image>:128815\n",
      "Add grounding-related tokens = ['<|ref|>', '<|/ref|>', '<|det|>', '<|/det|>', '<|grounding|>'] to the tokenizer with input_ids\n",
      "<|ref|>:128816\n",
      "<|/ref|>:128817\n",
      "<|det|>:128818\n",
      "<|/det|>:128819\n",
      "<|grounding|>:128820\n",
      "Add chat tokens = ['<|User|>', '<|Assistant|>'] to the tokenizer with input_ids\n",
      "<|User|>:128821\n",
      "<|Assistant|>:128822\n",
      "\n",
      "✅ Original model successfully loaded\n",
      "⌛ Convert Input embedding model\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:nncf:NNCF provides best results with torch==2.6.*, while current torch version is 2.2.2+cu121. If you encounter issues, consider switching to torch==2.6.*\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Input embedding model successfully converted"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "⌛ Convert Image embedding model\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.\n",
      "/environment/miniconda3/lib/python3.11/site-packages/torch/__init__.py:1499: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!\n",
      "  assert condition, message\n",
      "/environment/miniconda3/lib/python3.11/site-packages/nncf/torch/dynamic_graph/wrappers.py:85: TracerWarning: Converting a tensor to a Python float might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!\n",
      "  op1 = operator(*args, **kwargs)\n",
      "/home/featurize/DeepSeek-VL2/deepseek_vl2/models/modeling_deepseek_vl_v2.py:95: TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!\n",
      "  h = w = int((hw) ** 0.5)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "/environment/miniconda3/lib/python3.11/site-packages/openvino/runtime/__init__.py:10: DeprecationWarning: The `openvino.runtime` module is deprecated and will be removed in the 2026.0 release. Please replace `openvino.runtime` with `openvino`.\n",
      "  warnings.warn(\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:nncf:Statistics of the bitwidth distribution:\n",
      "┍━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑\n",
      "│ Weight compression mode   │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │\n",
      "┝━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥\n",
      "│ int8_asym                 │ 100% (111 / 111)            │ 100% (111 / 111)                       │\n",
      "┕━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">/environment/miniconda3/lib/python3.11/site-packages/rich/live.py:231: UserWarning: install \"ipywidgets\" for \n",
       "Jupyter support\n",
       "  warnings.warn('install \"ipywidgets\" for Jupyter support')\n",
       "</pre>\n"
      ],
      "text/plain": [
       "/environment/miniconda3/lib/python3.11/site-packages/rich/live.py:231: UserWarning: install \"ipywidgets\" for \n",
       "Jupyter support\n",
       "  warnings.warn('install \"ipywidgets\" for Jupyter support')\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"></pre>\n"
      ],
      "text/plain": []
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Image embedding model successfully converted\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/environment/miniconda3/lib/python3.11/site-packages/transformers/cache_utils.py:457: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!\n",
      "  or not self.key_cache[layer_idx].numel()  # the layer has no cache\n",
      "/environment/miniconda3/lib/python3.11/site-packages/transformers/modeling_attn_mask_utils.py:116: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!\n",
      "  if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:\n",
      "/environment/miniconda3/lib/python3.11/site-packages/transformers/modeling_attn_mask_utils.py:164: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!\n",
      "  if past_key_values_length > 0:\n",
      "/environment/miniconda3/lib/python3.11/site-packages/transformers/cache_utils.py:440: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!\n",
      "  elif (\n",
      "/environment/miniconda3/lib/python3.11/site-packages/nncf/torch/dynamic_graph/wrappers.py:85: TracerWarning: torch.tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.\n",
      "  op1 = operator(*args, **kwargs)\n",
      "/home/featurize/ov_deepseek_vl_helper.py:355: TracerWarning: Iterating over a tensor might cause the trace to be incorrect. Passing a tensor of different shape won't change the number of iterations executed (and might lead to errors or silently give incorrect results).\n",
      "  for i, num_tokens in enumerate(tokens_per_expert):\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Language model successfully converted\n",
      "⌛ Weights compression with int4_sym mode started\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:nncf:Statistics of the bitwidth distribution:\n",
      "┍━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑\n",
      "│ Weight compression mode   │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │\n",
      "┝━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥\n",
      "│ int8_asym                 │ 6% (1 / 2208)               │ 0% (0 / 2207)                          │\n",
      "├───────────────────────────┼─────────────────────────────┼────────────────────────────────────────┤\n",
      "│ int4_sym                  │ 94% (2207 / 2208)           │ 100% (2207 / 2207)                     │\n",
      "┕━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"></pre>\n"
      ],
      "text/plain": []
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Weights compression finished\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ deepseek-ai/deepseek-vl2-tiny model conversion finished. You can find results in deepseek-vl2-tiny/INT4\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "PosixPath('deepseek-vl2-tiny/INT4')"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 모델 준비\n",
    "from ov_deepseek_vl_helper import prepare_model_repo\n",
    "prepare_model_repo()\n",
    "\n",
    "# 변환에 필요한 설정\n",
    "from ov_deepseek_vl_helper import model_ids, convert_deepseek_vl\n",
    "from pathlib import Path\n",
    "import nncf\n",
    "\n",
    "model_id = \"deepseek-ai/deepseek-vl2-tiny\"  # 필요시 medium/large 등으로 변경 가능\n",
    "quantization_config = {\n",
    "    \"vision\": {\"mode\": nncf.CompressWeightsMode.INT8_ASYM},\n",
    "    \"llm\": {\n",
    "        \"mode\": nncf.CompressWeightsMode.INT4_SYM,\n",
    "        \"group_size\": 64 if \"tiny\" in model_id else -1,\n",
    "        \"ratio\": 1.0\n",
    "    },\n",
    "}\n",
    "\n",
    "model_path = Path(model_id.split(\"/\")[-1]) / \"INT4\"\n",
    "\n",
    "# 모델 변환 및 압축\n",
    "convert_deepseek_vl(model_id, model_path=model_path, quantization_config=quantization_config)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "73e948d9-3817-45a8-9f00-b53affac7a5e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#3단계. 추론 파이프라인 구성"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "cf82b876-51ed-448f-88e5-d3e4267a75ce",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2271095bc05c4af2a532b1b8f6dde5ff",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Add pad token = ['<｜▁pad▁｜>'] to the tokenizer\n",
      "<｜▁pad▁｜>:2\n",
      "Add image token = ['<image>'] to the tokenizer\n",
      "<image>:128815\n",
      "Add grounding-related tokens = ['<|ref|>', '<|/ref|>', '<|det|>', '<|/det|>', '<|grounding|>'] to the tokenizer with input_ids\n",
      "<|ref|>:128816\n",
      "<|/ref|>:128817\n",
      "<|det|>:128818\n",
      "<|/det|>:128819\n",
      "<|grounding|>:128820\n",
      "Add chat tokens = ['<|User|>', '<|Assistant|>'] to the tokenizer with input_ids\n",
      "<|User|>:128821\n",
      "<|Assistant|>:128822\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from ov_deepseek_vl_helper import OVDeepseekVLV2ForCausalLM\n",
    "from deepseek_vl2.models import DeepseekVLV2Processor\n",
    "from notebook_utils import device_widget\n",
    "\n",
    "# 장치 선택 (CPU 추천)\n",
    "device = device_widget(default=\"CPU\")\n",
    "display(device)\n",
    "\n",
    "# 모델 & 토크나이저 로딩\n",
    "ov_model = OVDeepseekVLV2ForCausalLM(model_path, device.value)\n",
    "processor = DeepseekVLV2Processor.from_pretrained(model_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1260ff26-f853-4eb7-b27e-73779c9a8c2e",
   "metadata": {},
   "outputs": [],
   "source": [
    "이건 내가 jupyter 재시작 하기 싫어서 파일 덮어씌우려고 만든거"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "6ca639eb-392a-475c-a0e8-75c439bf1ad9",
   "metadata": {},
   "outputs": [
    {
     "ename": "ImportError",
     "evalue": "Using SOCKS proxy, but the 'socksio' package is not installed. Make sure to install httpx using `pip install httpx[socks]`.",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mImportError\u001b[0m                               Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[59], line 10\u001b[0m\n\u001b[1;32m      8\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mimportlib\u001b[39;00m\n\u001b[1;32m      9\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mov_deepseek_vl_helper\u001b[39;00m\n\u001b[0;32m---> 10\u001b[0m \u001b[43mimportlib\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mreload\u001b[49m\u001b[43m(\u001b[49m\u001b[43mov_deepseek_vl_helper\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[0;32m/environment/miniconda3/lib/python3.11/importlib/__init__.py:169\u001b[0m, in \u001b[0;36mreload\u001b[0;34m(module)\u001b[0m\n\u001b[1;32m    167\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m spec \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[1;32m    168\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mModuleNotFoundError\u001b[39;00m(\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mspec not found for the module \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mname\u001b[38;5;132;01m!r}\u001b[39;00m\u001b[38;5;124m\"\u001b[39m, name\u001b[38;5;241m=\u001b[39mname)\n\u001b[0;32m--> 169\u001b[0m \u001b[43m_bootstrap\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_exec\u001b[49m\u001b[43m(\u001b[49m\u001b[43mspec\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mmodule\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    170\u001b[0m \u001b[38;5;66;03m# The module may have replaced itself in sys.modules!\u001b[39;00m\n\u001b[1;32m    171\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m sys\u001b[38;5;241m.\u001b[39mmodules[name]\n",
      "File \u001b[0;32m<frozen importlib._bootstrap>:621\u001b[0m, in \u001b[0;36m_exec\u001b[0;34m(spec, module)\u001b[0m\n",
      "File \u001b[0;32m<frozen importlib._bootstrap_external>:940\u001b[0m, in \u001b[0;36mexec_module\u001b[0;34m(self, module)\u001b[0m\n",
      "File \u001b[0;32m<frozen importlib._bootstrap>:241\u001b[0m, in \u001b[0;36m_call_with_frames_removed\u001b[0;34m(f, *args, **kwds)\u001b[0m\n",
      "File \u001b[0;32m~/ov_deepseek_vl_helper.py:18\u001b[0m\n\u001b[1;32m     16\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mre\u001b[39;00m\n\u001b[1;32m     17\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mPIL\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m ImageDraw, ImageFont\n\u001b[0;32m---> 18\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mdeepseek_vl2\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mserve\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mapp_modules\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mpresets\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m BOX2COLOR\n\u001b[1;32m     20\u001b[0m model_ids \u001b[38;5;241m=\u001b[39m [\n\u001b[1;32m     21\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mdeepseek-ai/deepseek-vl2-tiny\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     22\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mdeepseek-ai/deepseek-vl2-small\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     23\u001b[0m ]\n\u001b[1;32m     25\u001b[0m model_id \u001b[38;5;241m=\u001b[39m model_ids[\u001b[38;5;241m0\u001b[39m]\n",
      "File \u001b[0;32m~/DeepSeek-VL2/deepseek_vl2/serve/app_modules/presets.py:21\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[38;5;66;03m# Copyright (c) 2023-2024 DeepSeek.\u001b[39;00m\n\u001b[1;32m      2\u001b[0m \u001b[38;5;66;03m#\u001b[39;00m\n\u001b[1;32m      3\u001b[0m \u001b[38;5;66;03m# Permission is hereby granted, free of charge, to any person obtaining a copy of\u001b[39;00m\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m     19\u001b[0m \n\u001b[1;32m     20\u001b[0m \u001b[38;5;66;03m# -*- coding:utf-8 -*-\u001b[39;00m\n\u001b[0;32m---> 21\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mgradio\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mgr\u001b[39;00m\n\u001b[1;32m     23\u001b[0m title \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m\"\"\"\u001b[39m\u001b[38;5;124m<h1 align=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mleft\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m style=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmin-width:200px; margin-top:0;\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m>Chat with DeepSeek-VL2 </h1>\u001b[39m\u001b[38;5;124m\"\"\"\u001b[39m\n\u001b[1;32m     24\u001b[0m description_top \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m\"\"\"\u001b[39m\u001b[38;5;124mSpecial Tokens: `<image>`,     Visual Grounding: `<|ref|>\u001b[39m\u001b[38;5;132;01m{query}\u001b[39;00m\u001b[38;5;124m<|/ref|>`,    Grounding Conversation: `<|grounding|>\u001b[39m\u001b[38;5;132;01m{question}\u001b[39;00m\u001b[38;5;124m`\u001b[39m\u001b[38;5;124m\"\"\"\u001b[39m\n",
      "File \u001b[0;32m/environment/miniconda3/lib/python3.11/site-packages/gradio/__init__.py:3\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mjson\u001b[39;00m\n\u001b[0;32m----> 3\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mgradio\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01m_simple_templates\u001b[39;00m\n\u001b[1;32m      4\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mgradio\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mimage_utils\u001b[39;00m\n\u001b[1;32m      5\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mgradio\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mprocessing_utils\u001b[39;00m\n",
      "File \u001b[0;32m/environment/miniconda3/lib/python3.11/site-packages/gradio/_simple_templates/__init__.py:1\u001b[0m\n\u001b[0;32m----> 1\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01msimpledropdown\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m SimpleDropdown\n\u001b[1;32m      2\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01msimpleimage\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m SimpleImage\n\u001b[1;32m      3\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01msimpletextbox\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m SimpleTextbox\n",
      "File \u001b[0;32m/environment/miniconda3/lib/python3.11/site-packages/gradio/_simple_templates/simpledropdown.py:7\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mcollections\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mabc\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m Callable, Sequence\n\u001b[1;32m      5\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mtyping\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m TYPE_CHECKING, Any\n\u001b[0;32m----> 7\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mgradio\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mcomponents\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mbase\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m Component, FormComponent\n\u001b[1;32m      8\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mgradio\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mevents\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m Events\n\u001b[1;32m     10\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m TYPE_CHECKING:\n",
      "File \u001b[0;32m/environment/miniconda3/lib/python3.11/site-packages/gradio/components/__init__.py:1\u001b[0m\n\u001b[0;32m----> 1\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mgradio\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mcomponents\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mannotated_image\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m AnnotatedImage\n\u001b[1;32m      2\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mgradio\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mcomponents\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01maudio\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m Audio\n\u001b[1;32m      3\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mgradio\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mcomponents\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mbase\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m (\n\u001b[1;32m      4\u001b[0m     Component,\n\u001b[1;32m      5\u001b[0m     FormComponent,\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m     10\u001b[0m     get_component_instance,\n\u001b[1;32m     11\u001b[0m )\n",
      "File \u001b[0;32m/environment/miniconda3/lib/python3.11/site-packages/gradio/components/annotated_image.py:14\u001b[0m\n\u001b[1;32m     11\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mgradio_client\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m handle_file\n\u001b[1;32m     12\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mgradio_client\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mdocumentation\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m document\n\u001b[0;32m---> 14\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mgradio\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m processing_utils, utils\n\u001b[1;32m     15\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mgradio\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mcomponents\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mbase\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m Component\n\u001b[1;32m     16\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mgradio\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mdata_classes\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m FileData, GradioModel\n",
      "File \u001b[0;32m/environment/miniconda3/lib/python3.11/site-packages/gradio/processing_utils.py:120\u001b[0m\n\u001b[1;32m    117\u001b[0m     sync_transport \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[1;32m    118\u001b[0m     async_transport \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[0;32m--> 120\u001b[0m sync_client \u001b[38;5;241m=\u001b[39m \u001b[43mhttpx\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mClient\u001b[49m\u001b[43m(\u001b[49m\u001b[43mtransport\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43msync_transport\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    122\u001b[0m log \u001b[38;5;241m=\u001b[39m logging\u001b[38;5;241m.\u001b[39mgetLogger(\u001b[38;5;18m__name__\u001b[39m)\n\u001b[1;32m    124\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m TYPE_CHECKING:\n",
      "File \u001b[0;32m/environment/miniconda3/lib/python3.11/site-packages/httpx/_client.py:695\u001b[0m, in \u001b[0;36mClient.__init__\u001b[0;34m(self, auth, params, headers, cookies, verify, cert, http1, http2, proxy, proxies, mounts, timeout, follow_redirects, limits, max_redirects, event_hooks, base_url, transport, app, trust_env, default_encoding)\u001b[0m\n\u001b[1;32m    683\u001b[0m proxy_map \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_get_proxy_map(proxies \u001b[38;5;129;01mor\u001b[39;00m proxy, allow_env_proxies)\n\u001b[1;32m    685\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_transport \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_init_transport(\n\u001b[1;32m    686\u001b[0m     verify\u001b[38;5;241m=\u001b[39mverify,\n\u001b[1;32m    687\u001b[0m     cert\u001b[38;5;241m=\u001b[39mcert,\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m    693\u001b[0m     trust_env\u001b[38;5;241m=\u001b[39mtrust_env,\n\u001b[1;32m    694\u001b[0m )\n\u001b[0;32m--> 695\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_mounts: \u001b[38;5;28mdict\u001b[39m[URLPattern, BaseTransport \u001b[38;5;241m|\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m] \u001b[38;5;241m=\u001b[39m \u001b[43m{\u001b[49m\n\u001b[1;32m    696\u001b[0m \u001b[43m    \u001b[49m\u001b[43mURLPattern\u001b[49m\u001b[43m(\u001b[49m\u001b[43mkey\u001b[49m\u001b[43m)\u001b[49m\u001b[43m:\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;28;43;01mNone\u001b[39;49;00m\n\u001b[1;32m    697\u001b[0m \u001b[43m    \u001b[49m\u001b[38;5;28;43;01mif\u001b[39;49;00m\u001b[43m \u001b[49m\u001b[43mproxy\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;129;43;01mis\u001b[39;49;00m\u001b[43m \u001b[49m\u001b[38;5;28;43;01mNone\u001b[39;49;00m\n\u001b[1;32m    698\u001b[0m \u001b[43m    \u001b[49m\u001b[38;5;28;43;01melse\u001b[39;49;00m\u001b[43m \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_init_proxy_transport\u001b[49m\u001b[43m(\u001b[49m\n\u001b[1;32m    699\u001b[0m \u001b[43m        \u001b[49m\u001b[43mproxy\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    700\u001b[0m \u001b[43m        \u001b[49m\u001b[43mverify\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mverify\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    701\u001b[0m \u001b[43m        \u001b[49m\u001b[43mcert\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mcert\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    702\u001b[0m \u001b[43m        \u001b[49m\u001b[43mhttp1\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mhttp1\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    703\u001b[0m \u001b[43m        \u001b[49m\u001b[43mhttp2\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mhttp2\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    704\u001b[0m \u001b[43m        \u001b[49m\u001b[43mlimits\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mlimits\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    705\u001b[0m \u001b[43m        \u001b[49m\u001b[43mtrust_env\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mtrust_env\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    706\u001b[0m \u001b[43m    \u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    707\u001b[0m \u001b[43m    \u001b[49m\u001b[38;5;28;43;01mfor\u001b[39;49;00m\u001b[43m \u001b[49m\u001b[43mkey\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mproxy\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;129;43;01min\u001b[39;49;00m\u001b[43m \u001b[49m\u001b[43mproxy_map\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mitems\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    708\u001b[0m \u001b[43m\u001b[49m\u001b[43m}\u001b[49m\n\u001b[1;32m    709\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m mounts \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[1;32m    710\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_mounts\u001b[38;5;241m.\u001b[39mupdate(\n\u001b[1;32m    711\u001b[0m         {URLPattern(key): transport \u001b[38;5;28;01mfor\u001b[39;00m key, transport \u001b[38;5;129;01min\u001b[39;00m mounts\u001b[38;5;241m.\u001b[39mitems()}\n\u001b[1;32m    712\u001b[0m     )\n",
      "File \u001b[0;32m/environment/miniconda3/lib/python3.11/site-packages/httpx/_client.py:698\u001b[0m, in \u001b[0;36m<dictcomp>\u001b[0;34m(.0)\u001b[0m\n\u001b[1;32m    683\u001b[0m proxy_map \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_get_proxy_map(proxies \u001b[38;5;129;01mor\u001b[39;00m proxy, allow_env_proxies)\n\u001b[1;32m    685\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_transport \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_init_transport(\n\u001b[1;32m    686\u001b[0m     verify\u001b[38;5;241m=\u001b[39mverify,\n\u001b[1;32m    687\u001b[0m     cert\u001b[38;5;241m=\u001b[39mcert,\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m    693\u001b[0m     trust_env\u001b[38;5;241m=\u001b[39mtrust_env,\n\u001b[1;32m    694\u001b[0m )\n\u001b[1;32m    695\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_mounts: \u001b[38;5;28mdict\u001b[39m[URLPattern, BaseTransport \u001b[38;5;241m|\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m] \u001b[38;5;241m=\u001b[39m {\n\u001b[1;32m    696\u001b[0m     URLPattern(key): \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[1;32m    697\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m proxy \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[0;32m--> 698\u001b[0m     \u001b[38;5;28;01melse\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_init_proxy_transport\u001b[49m\u001b[43m(\u001b[49m\n\u001b[1;32m    699\u001b[0m \u001b[43m        \u001b[49m\u001b[43mproxy\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    700\u001b[0m \u001b[43m        \u001b[49m\u001b[43mverify\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mverify\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    701\u001b[0m \u001b[43m        \u001b[49m\u001b[43mcert\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mcert\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    702\u001b[0m \u001b[43m        \u001b[49m\u001b[43mhttp1\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mhttp1\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    703\u001b[0m \u001b[43m        \u001b[49m\u001b[43mhttp2\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mhttp2\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    704\u001b[0m \u001b[43m        \u001b[49m\u001b[43mlimits\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mlimits\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    705\u001b[0m \u001b[43m        \u001b[49m\u001b[43mtrust_env\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mtrust_env\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    706\u001b[0m \u001b[43m    \u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    707\u001b[0m     \u001b[38;5;28;01mfor\u001b[39;00m key, proxy \u001b[38;5;129;01min\u001b[39;00m proxy_map\u001b[38;5;241m.\u001b[39mitems()\n\u001b[1;32m    708\u001b[0m }\n\u001b[1;32m    709\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m mounts \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[1;32m    710\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_mounts\u001b[38;5;241m.\u001b[39mupdate(\n\u001b[1;32m    711\u001b[0m         {URLPattern(key): transport \u001b[38;5;28;01mfor\u001b[39;00m key, transport \u001b[38;5;129;01min\u001b[39;00m mounts\u001b[38;5;241m.\u001b[39mitems()}\n\u001b[1;32m    712\u001b[0m     )\n",
      "File \u001b[0;32m/environment/miniconda3/lib/python3.11/site-packages/httpx/_client.py:752\u001b[0m, in \u001b[0;36mClient._init_proxy_transport\u001b[0;34m(self, proxy, verify, cert, http1, http2, limits, trust_env)\u001b[0m\n\u001b[1;32m    742\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m_init_proxy_transport\u001b[39m(\n\u001b[1;32m    743\u001b[0m     \u001b[38;5;28mself\u001b[39m,\n\u001b[1;32m    744\u001b[0m     proxy: Proxy,\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m    750\u001b[0m     trust_env: \u001b[38;5;28mbool\u001b[39m \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mTrue\u001b[39;00m,\n\u001b[1;32m    751\u001b[0m ) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m BaseTransport:\n\u001b[0;32m--> 752\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mHTTPTransport\u001b[49m\u001b[43m(\u001b[49m\n\u001b[1;32m    753\u001b[0m \u001b[43m        \u001b[49m\u001b[43mverify\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mverify\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    754\u001b[0m \u001b[43m        \u001b[49m\u001b[43mcert\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mcert\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    755\u001b[0m \u001b[43m        \u001b[49m\u001b[43mhttp1\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mhttp1\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    756\u001b[0m \u001b[43m        \u001b[49m\u001b[43mhttp2\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mhttp2\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    757\u001b[0m \u001b[43m        \u001b[49m\u001b[43mlimits\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mlimits\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    758\u001b[0m \u001b[43m        \u001b[49m\u001b[43mtrust_env\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mtrust_env\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    759\u001b[0m \u001b[43m        \u001b[49m\u001b[43mproxy\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mproxy\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    760\u001b[0m \u001b[43m    \u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[0;32m/environment/miniconda3/lib/python3.11/site-packages/httpx/_transports/default.py:175\u001b[0m, in \u001b[0;36mHTTPTransport.__init__\u001b[0;34m(self, verify, cert, http1, http2, limits, trust_env, proxy, uds, local_address, retries, socket_options)\u001b[0m\n\u001b[1;32m    173\u001b[0m         \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01msocksio\u001b[39;00m  \u001b[38;5;66;03m# noqa\u001b[39;00m\n\u001b[1;32m    174\u001b[0m     \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mImportError\u001b[39;00m:  \u001b[38;5;66;03m# pragma: no cover\u001b[39;00m\n\u001b[0;32m--> 175\u001b[0m         \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mImportError\u001b[39;00m(\n\u001b[1;32m    176\u001b[0m             \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mUsing SOCKS proxy, but the \u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124msocksio\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m package is not installed. \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    177\u001b[0m             \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mMake sure to install httpx using `pip install httpx[socks]`.\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    178\u001b[0m         ) \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[1;32m    180\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_pool \u001b[38;5;241m=\u001b[39m httpcore\u001b[38;5;241m.\u001b[39mSOCKSProxy(\n\u001b[1;32m    181\u001b[0m         proxy_url\u001b[38;5;241m=\u001b[39mhttpcore\u001b[38;5;241m.\u001b[39mURL(\n\u001b[1;32m    182\u001b[0m             scheme\u001b[38;5;241m=\u001b[39mproxy\u001b[38;5;241m.\u001b[39murl\u001b[38;5;241m.\u001b[39mraw_scheme,\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m    193\u001b[0m         http2\u001b[38;5;241m=\u001b[39mhttp2,\n\u001b[1;32m    194\u001b[0m     )\n\u001b[1;32m    195\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:  \u001b[38;5;66;03m# pragma: no cover\u001b[39;00m\n",
      "\u001b[0;31mImportError\u001b[0m: Using SOCKS proxy, but the 'socksio' package is not installed. Make sure to install httpx using `pip install httpx[socks]`."
     ]
    }
   ],
   "source": [
    "import requests\n",
    "\n",
    "url = \"https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/deepseek-vl2/ov_deepseek_vl_helper.py\"\n",
    "r = requests.get(url)\n",
    "\n",
    "with open(\"ov_deepseek_vl_helper.py\", \"w\", encoding=\"utf-8\") as f:\n",
    "    f.write(r.text)\n",
    "import importlib\n",
    "import ov_deepseek_vl_helper\n",
    "importlib.reload(ov_deepseek_vl_helper)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "826b01d5-945c-42ca-ae7d-85e4e25a00fd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ parse_ref_bbox in file: True\n"
     ]
    }
   ],
   "source": [
    "# 👀 ov_deepseek_vl_helper.py 파일 내용 확인\n",
    "with open(\"ov_deepseek_vl_helper.py\") as f:\n",
    "    content = f.read()\n",
    "    print(\"✅ parse_ref_bbox in file:\", \"parse_ref_bbox\" in content)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "586551ce-6126-4c8d-a065-cdace756530b",
   "metadata": {},
   "outputs": [],
   "source": [
    "#4단계. 이미지 기반 질의응답 테스트"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "bcb6d379-1602-44e4-a9f9-31b3b8cc61b7",
   "metadata": {},
   "outputs": [
    {
     "ename": "ImportError",
     "evalue": "cannot import name 'parse_ref_bbox' from 'ov_deepseek_vl_helper' (/home/featurize/ov_deepseek_vl_helper.py)",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mImportError\u001b[0m                               Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[62], line 3\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mPIL\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m Image\n\u001b[1;32m      2\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mtransformers\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m TextStreamer\n\u001b[0;32m----> 3\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mov_deepseek_vl_helper\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m parse_ref_bbox\n\u001b[1;32m      5\u001b[0m \u001b[38;5;66;03m# 예제 대화 설정\u001b[39;00m\n\u001b[1;32m      6\u001b[0m conversation \u001b[38;5;241m=\u001b[39m [\n\u001b[1;32m      7\u001b[0m     {\n\u001b[1;32m      8\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mrole\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m<|User|>\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m     12\u001b[0m     {\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mrole\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m<|Assistant|>\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcontent\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m\"\u001b[39m},\n\u001b[1;32m     13\u001b[0m ]\n",
      "\u001b[0;31mImportError\u001b[0m: cannot import name 'parse_ref_bbox' from 'ov_deepseek_vl_helper' (/home/featurize/ov_deepseek_vl_helper.py)"
     ]
    }
   ],
   "source": [
    "from PIL import Image\n",
    "from transformers import TextStreamer\n",
    "from ov_deepseek_vl_helper import parse_ref_bbox\n",
    "\n",
    "# 예제 대화 설정\n",
    "conversation = [\n",
    "    {\n",
    "        \"role\": \"<|User|>\",\n",
    "        \"content\": \"<image>\\n<|ref|>The giraffe at the back.<|/ref|>\",\n",
    "        \"images\": [\"DeepSeek-VL2/images/visual_grounding_1.jpeg\"],\n",
    "    },\n",
    "    {\"role\": \"<|Assistant|>\", \"content\": \"\"},\n",
    "]\n",
    "\n",
    "# 이미지 불러오기\n",
    "def load_pil_images(conversations):\n",
    "    pil_images = []\n",
    "    for message in conversations:\n",
    "        if \"images\" not in message:\n",
    "            continue\n",
    "        for image_path in message[\"images\"]:\n",
    "            pil_img = Image.open(image_path).convert(\"RGB\")\n",
    "            pil_images.append(pil_img)\n",
    "    return pil_images\n",
    "\n",
    "pil_images = load_pil_images(conversation)\n",
    "\n",
    "# 입력 전처리\n",
    "prepare_inputs = processor(conversations=conversation, images=pil_images, force_batchify=True, system_prompt=\"\")\n",
    "\n",
    "# 추론\n",
    "inputs_embeds = ov_model.prepare_inputs_embeds(**prepare_inputs)\n",
    "outputs = ov_model.language_model.generate(\n",
    "    inputs_embeds=inputs_embeds,\n",
    "    attention_mask=None,\n",
    "    pad_token_id=processor.tokenizer.eos_token_id,\n",
    "    bos_token_id=processor.tokenizer.bos_token_id,\n",
    "    eos_token_id=processor.tokenizer.eos_token_id,\n",
    "    max_new_tokens=512,\n",
    "    do_sample=False,\n",
    "    use_cache=True,\n",
    "    streamer=TextStreamer(processor.tokenizer, skip_special_tokens=True),\n",
    ")\n",
    "\n",
    "# 결과 디코딩\n",
    "answer = processor.tokenizer.batch_decode(outputs, skip_special_tokens=False)\n",
    "vg_image = parse_ref_bbox(answer[0], image=pil_images[-1])\n",
    "if vg_image is not None:\n",
    "    display(vg_image)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e57b5079-ba30-4260-aa72-1eff7d04c394",
   "metadata": {},
   "outputs": [],
   "source": [
    "여기부터는 보지 마셈. 내가 테스트 해보고 싶어서 한거"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "40cd002c-503f-4851-af5d-178e968d3374",
   "metadata": {},
   "outputs": [],
   "source": [
    "import re\n",
    "from PIL import ImageDraw, ImageFont\n",
    "\n",
    "BOX2COLOR = [\n",
    "    \"red\", \"blue\", \"green\", \"orange\", \"purple\", \"brown\", \"cyan\", \"magenta\",\n",
    "    \"lime\", \"yellow\", \"pink\", \"teal\", \"gold\", \"black\"\n",
    "]\n",
    "\n",
    "def parse_ref_bbox(response, image):\n",
    "    try:\n",
    "        image = image.copy()\n",
    "        image_w, image_h = image.size\n",
    "        draw = ImageDraw.Draw(image)\n",
    "\n",
    "        ref = re.findall(r\"<\\|ref\\|>.*?<\\|/ref\\|>\", response)\n",
    "        bbox = re.findall(r\"<\\|det\\|>.*?<\\|/det\\|>\", response)\n",
    "\n",
    "        if len(ref) != len(bbox):\n",
    "            return None\n",
    "\n",
    "        boxes, labels = [], []\n",
    "        for box, label in zip(bbox, ref):\n",
    "            box = box.replace(\"<|det|>\", \"\").replace(\"<|/det|>\", \"\")\n",
    "            label = label.replace(\"<|ref|>\", \"\").replace(\"<|/ref|>\", \"\")\n",
    "            for onebox in re.findall(r\"\\[.*?\\]\", box):\n",
    "                boxes.append(eval(onebox))\n",
    "                labels.append(label)\n",
    "\n",
    "        for i, (box, label) in enumerate(zip(boxes, labels)):\n",
    "            box = (\n",
    "                int(box[0] / 999 * image_w),\n",
    "                int(box[1] / 999 * image_h),\n",
    "                int(box[2] / 999 * image_w),\n",
    "                int(box[3] / 999 * image_h),\n",
    "            )\n",
    "            color = BOX2COLOR[i % len(BOX2COLOR)]\n",
    "            draw.rectangle(box, outline=color, width=2)\n",
    "\n",
    "            try:\n",
    "                font = ImageFont.truetype(\"DeepSeek-VL2/deepseek_vl2/serve/assets/simsun.ttc\", size=20)\n",
    "            except:\n",
    "                font = None\n",
    "\n",
    "            draw.text((box[0], box[1] - 20), label, fill=color, font=font)\n",
    "\n",
    "        return image\n",
    "    except Exception as e:\n",
    "        print(\"🔴 parse_ref_bbox 오류:\", e)\n",
    "        return None\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "e1b20089-4522-4570-a53e-841b92bef196",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🔴 parse_ref_bbox 오류: '[' was never closed (<string>, line 1)\n"
     ]
    },
    {
     "ename": "AttributeError",
     "evalue": "'NoneType' object has no attribute 'show'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[66], line 8\u001b[0m\n\u001b[1;32m      5\u001b[0m response \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m<|ref|>쓰레기통<|/ref|><|det|>[[200, 300, 400, 500]]<|/det|>\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m      7\u001b[0m result_img \u001b[38;5;241m=\u001b[39m parse_ref_bbox(response, image)\n\u001b[0;32m----> 8\u001b[0m \u001b[43mresult_img\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mshow\u001b[49m()  \u001b[38;5;66;03m# 또는 IPython.display로 출력\u001b[39;00m\n",
      "\u001b[0;31mAttributeError\u001b[0m: 'NoneType' object has no attribute 'show'"
     ]
    }
   ],
   "source": [
    "# 예시: 이미지와 응답 문자열을 가지고 시각화\n",
    "from PIL import Image\n",
    "\n",
    "image = Image.open(\"DeepSeek-VL2/images/visual_grounding_1.jpeg\")\n",
    "response = \"<|ref|>쓰레기통<|/ref|><|det|>[[200, 300, 400, 500]]<|/det|>\"\n",
    "\n",
    "result_img = parse_ref_bbox(response, image)\n",
    "result_img.show()  # 또는 IPython.display로 출력\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1cc57cdd-a1a1-4556-9620-8a9c9da6342e",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
