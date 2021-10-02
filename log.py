import os
import sys
import time
import zipfile
from tensorboardX import SummaryWriter
projectDir = os.path.dirname(__file__)


def findExt(folder, extensions, exclude_list):
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        if any(substring in root for substring in exclude_list):
            continue
        for extension in extensions:
            for filename in filenames:
                if filename.endswith(extension):
                    matches.append(os.path.join(root, filename))
    return matches


def backup_code(outfname, folder, extensions, exclude_list):
    filenames = findExt(folder, extensions, exclude_list)
    zf = zipfile.ZipFile(outfname, mode='w')
    for filename in filenames:
        zf.write(filename)
    zf.close()
    print('saved %i files to %s' % (len(filenames), outfname))

class logger_tb(object):

    def __init__(self, log_dir, description, code_backup, log_tb, time_stamp=True):
        """Create a summary writer logging to log_dir."""
        log_dir = log_dir
        desc = description.split(" ")
        if time_stamp == True:
            t_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            desc.insert(0, t_stamp)
            self.m_dir = "_".join(desc)
            log_dir = os.path.join(log_dir, self.m_dir)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        log_dir = os.path.join(projectDir, log_dir)
        if log_tb:
            self.writer = SummaryWriter(log_dir)
        self.g_step = 0
        self.graph_log = 1
        if code_backup:
            backup_code(os.path.join(log_dir, 'code_snapshot.zip'), '.', ['.py', ".pyc",'.json', '.sh',".txt",".cpp",".so",".o",".cu"], ['checkpoints', 'logs', 'data',"dataset"])

    def log_dir(self):
        return self.m_dir

    def scalar_summary(self, tag, value):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, self.g_step)
        self.writer.flush()

    def image_summary(self, tag, image):
        """Log a list of images."""
        self.writer.add_image(tag, image, self.g_step)
        self.writer.flush()


    def histo_summary(self, tag, values):
        """Log a histogram of the tensor of values."""
        # Create and write Summary
        self.writer.add_histogram(tag, values, self.g_step)
        self.writer.flush()

    def graph_summary(self, model, images):
        self.writer.add_graph(model, images)
        self.writer.flush()


class message_logger(object):
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(os.path.join(path, "log.txt"), 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass
