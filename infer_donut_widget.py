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

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_donut.infer_donut_process import InferDonutParam

# PyQt GUI framework
from PyQt5.QtWidgets import *

from infer_donut.model_zoo import model_zoo
from torch import cuda


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferDonutWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferDonutParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # Model
        self.combo_model_name = pyqtutils.append_combo(self.grid_layout, "Model")
        for model in model_zoo:
            self.combo_model_name.addItem(model)
        self.combo_model_name.setCurrentText(self.parameters.model_name)

        # Prompt
        self.edit_prompt = pyqtutils.append_edit(self.grid_layout, "Prompt", self.parameters.prompt)

        # Cuda
        self.check_cuda = pyqtutils.append_check(self.grid_layout, "Cuda", self.parameters.cuda and cuda.is_available())
        self.check_cuda.setEnabled(cuda.is_available())

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_apply(self):
        # Apply button clicked slot

        # Get parameters from widget
        # Example : self.parameters.windowSize = self.spinWindowSize.value()
        self.parameters.prompt = self.edit_prompt.text()
        is_cuda = self.check_cuda.isChecked()
        model_name = self.combo_model_name.currentText()
        if model_name in model_zoo:
            task_name = model_zoo[model_name]
        else:
            raise NotImplementedError
        if model_name != self.parameters.model_name or task_name != self.parameters.task_name or \
                is_cuda != self.parameters.cuda:
            self.parameters.update = True
        self.parameters.model_name = model_name
        self.parameters.cuda = is_cuda
        self.parameters.task_name = task_name
        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferDonutWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_donut"

    def create(self, param):
        # Create widget object
        return InferDonutWidget(param, None)
