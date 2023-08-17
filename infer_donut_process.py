# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import copy
from ikomia import core, dataprocess
from ikomia.utils import strtobool
from infer_donut.model import DonutModel
import torch
from PIL import Image
from infer_donut.model_zoo import model_zoo


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferDonutParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        # Example : self.windowSize = 25
        self.model_name = 'naver-clova-ix/donut-base-finetuned-docvqa'
        self.task_name = ''
        self.cuda = True
        self.prompt = 'what is the title'
        # used only with Ikomia STUDIO to store custom train browser's content
        self.browse_memory = ''

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.windowSize = int(param_map["windowSize"])
        if self.model_name != param_map["model_name"] or self.task_name != param_map["task_name"] or \
                self.cuda != strtobool(param_map["cuda"]):
            self.update = True
        self.model_name = param_map["model_name"]
        self.task_name = param_map["task_name"]
        self.prompt = param_map["prompt"]
        self.cuda = strtobool(param_map["cuda"])
        self.browse_memory = param_map["browse_memory"]

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        # Example : paramMap["windowSize"] = str(self.windowSize)
        param_map["model_name"] = self.model_name
        param_map["task_name"] = self.task_name
        param_map["prompt"] = self.prompt
        param_map["cuda"] = str(self.cuda)
        param_map["browse_memory"] = self.browse_memory
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferDonut(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the process here
        # Example :  self.add_input(dataprocess.CImageIO())
        #           self.add_output(dataprocess.CImageIO())

        # Create parameters class
        self.model = None
        self.add_output(dataprocess.DataDictIO())

        if param is None:
            self.set_param_object(InferDonutParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def infer(self, img, task_name, question):
        img = Image.fromarray(img)
        if task_name == "docvqa":
            output = self.model.inference(
                image=img,
                prompt=f"<s_{task_name}><s_question>{question.lower()}</s_question><s_answer>",
            )["predictions"][0]
        else:
            output = \
            self.model.inference(image=img, prompt=f"<s_{task_name}>")["predictions"][0]
        return output

    def run(self):
        # Core function of your process
        # Call begin_task_run() for initialization
        self.begin_task_run()

        param = self.get_param_object()

        if self.model is None or param.update:
            print("Loading model...")
            self.model = DonutModel.from_pretrained(param.model_name, ignore_mismatched_sizes=True)
            print("Model loaded.")
            if torch.cuda.is_available() and param.cuda:
                self.model.half()
                self.model.to("cuda")
            self.model.eval()

            if param.model_name in model_zoo:
                param.task_name = model_zoo[param.model_name]
            if param.task_name != 'docvqa' and param.prompt != '':
                print("Parameter prompt is only available for document visual question answering task.")
            param.update = False

        img_input = self.get_input(0)

        img = img_input.get_image()

        data_output = self.get_output(1)

        with torch.no_grad():
            data_output.data = self.infer(img, param.task_name, param.prompt)

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferDonutFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_donut"
        self.info.short_description = "your short description"
        self.info.description = "your description"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python"
        self.info.version = "1.0.0"
        # self.info.icon_path = "your path to a specific icon"
        self.info.authors = "algorithm author"
        self.info.article = "title of associated research article"
        self.info.journal = "publication journal"
        self.info.year = 2021
        self.info.license = "MIT License"
        # URL of documentation
        self.info.documentation_link = ""
        # Code source repository
        self.info.repository = ""
        # Keywords used for search
        self.info.keywords = "your,keywords,here"

    def create(self, param=None):
        # Create process object
        return InferDonut(self.info.name, param)
