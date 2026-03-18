# PyQt GUI framework
from PyQt6.QtWidgets import *

from torch import cuda

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion

from infer_donut.infer_donut_process import InferDonutParam
from infer_donut.model_zoo import model_zoo


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

        # Save current state
        self.cuda = self.parameters.cuda
        self.model_name = self.parameters.model_name

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # Pretrained model
        self.combo_model_name = pyqtutils.append_combo(self.grid_layout, "Pretrained model name")
        for model in model_zoo:
            self.combo_model_name.addItem(model)

        self.combo_model_name.setCurrentText(self.parameters.model_name)

        # Custom model
        self.browse_model_name = pyqtutils.append_browse_file(self.grid_layout, "Custom train folder",
                                                              self.parameters.custom_model_folder,
                                                              mode=QFileDialog.FileMode.Directory)

        # Prompt
        self.edit_prompt = pyqtutils.append_edit(self.grid_layout, "Prompt", self.parameters.prompt)

        # Task name
        self.edit_task_name = pyqtutils.append_edit(self.grid_layout,
                                                    "Task name (for custom train)",
                                                    self.parameters.task_name)

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
        self.parameters.prompt = self.edit_prompt.text()
        self.parameters.cuda = self.check_cuda.isChecked()
        model_name_input = self.browse_model_name.path

        if model_name_input != '':
            self.parameters.model_name = model_name_input
            self.parameters.task_name = self.edit_task_name.text()
            self.parameters.custom_model_folder = model_name_input
        else:
            self.parameters.model_name = self.combo_model_name.currentText()
            self.parameters.task_name = model_zoo[self.parameters.model_name]

        # Check state changes
        if self.parameters.model_name != self.model_name or self.parameters.cuda != self.cuda:
            self.model_name = self.parameters.model_name
            self.cuda = self.parameters.cuda
            self.parameters.update = True

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
