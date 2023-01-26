#==================================================================
#AESTHETIC UI CLASSES, METHODS, AND INITIAL SETUP
#==================================================================

#Quick print for titles in UI 
def sectionTitle(title):
    print()
    print((' ' * int((int(consoleColumns)-len(title))//2))+title)
    print(('#' * int(consoleColumns)))

#Construct and print a header for the running configuration
def programTitle(versionNum, configFileName):
    configHeader = os.path.splitext(os.path.basename(configFileName))[0]
    licenseHeader = "Licensed under GNU General Public License - v3.0"
    if erdModel == 'SLADS-LS': 
        programName = "Supervised Learning Approach for Dynamic Sampling with Least Squares"
        headerWidth = 65
    elif erdModel == 'SLADS-Net': 
        programName = "Supervised Learning Approach for Dynamic Sampling with Neural Network"
        headerWidth = 76
    elif erdModel == 'DLADS': 
        programName = "Deep Learning Approach for Dynamic Sampling"
        headerWidth = 40
    elif erdModel == 'GLANDS': 
        programName = "Generative Learning Adversarial Network for Dynamic Sampling"
        headerWidth = 50
    programName += " - v"+versionNum
    programNameOffset = (' ' * int((int(consoleColumns)-len(programName))//2))
    headerLineOffset = (' ' * int((int(consoleColumns)-headerWidth)//2))
    licenseOffset = (' ' * int((int(consoleColumns)-len(licenseHeader))//2))
    configOffset = (' ' * int((int(consoleColumns)-len(configHeader))//2))
    
    #Font: Dark with Shadow
    if erdModel == 'SLADS-LS': 
        header = "\n\
"+headerLineOffset+"███████╗██╗      █████╗ ██████╗ ███████╗         ██╗     ███████╗\n\
"+headerLineOffset+"██╔════╝██║     ██╔══██╗██╔══██╗██╔════╝         ██║     ██╔════╝\n\
"+headerLineOffset+"███████╗██║     ███████║██║  ██║███████╗ ███████ ██║     ███████╗\n\
"+headerLineOffset+"╚════██║██║     ██╔══██║██║  ██║╚════██║         ██║     ╚════██║\n\
"+headerLineOffset+"███████║███████╗██║  ██║██████╔╝███████║         ███████╗███████║\n\
"+headerLineOffset+"╚══════╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚══════╝         ╚══════╝╚══════╝\n"

    elif erdModel == 'SLADS-Net': 
        header = "\n\
"+headerLineOffset+"███████╗██╗      █████╗ ██████╗ ███████╗         ███╗   ██╗███████╗████████╗\n\
"+headerLineOffset+"██╔════╝██║     ██╔══██╗██╔══██╗██╔════╝         ████╗  ██║██╔════╝╚══██╔══╝\n\
"+headerLineOffset+"███████╗██║     ███████║██║  ██║███████╗ ███████ ██╔██╗ ██║█████╗     ██║   \n\
"+headerLineOffset+"╚════██║██║     ██╔══██║██║  ██║╚════██║         ██║╚██╗██║██╔══╝     ██║   \n\
"+headerLineOffset+"███████║███████╗██║  ██║██████╔╝███████║         ██║ ╚████║███████╗   ██║   \n\
"+headerLineOffset+"╚══════╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚══════╝         ╚═╝  ╚═══╝╚══════╝   ╚═╝   \n"

    elif erdModel == 'DLADS': 
        header = "\n\
"+headerLineOffset+"██████╗ ██╗      █████╗ ██████╗ ███████╗\n\
"+headerLineOffset+"██╔══██╗██║     ██╔══██╗██╔══██╗██╔════╝\n\
"+headerLineOffset+"██║  ██║██║     ███████║██║  ██║███████╗\n\
"+headerLineOffset+"██║  ██║██║     ██╔══██║██║  ██║╚════██║\n\
"+headerLineOffset+"██████╔╝███████╗██║  ██║██████╔╝███████║\n\
"+headerLineOffset+"╚═════╝ ╚══════╝╚═╝  ╚═╝╚═════╝ ╚══════╝\n"

    elif erdModel == 'GLANDS': 
        header = "\n\
"+headerLineOffset+" ██████╗ ██╗      █████╗ ███╗   ██╗██████╗ ███████╗\n\
"+headerLineOffset+"██╔════╝ ██║     ██╔══██╗████╗  ██║██╔══██╗██╔════╝\n\
"+headerLineOffset+"██║  ███╗██║     ███████║██╔██╗ ██║██║  ██║███████╗\n\
"+headerLineOffset+"██║   ██║██║     ██╔══██║██║╚██╗██║██║  ██║╚════██║\n\
"+headerLineOffset+"╚██████╔╝███████╗██║  ██║██║ ╚████║██████╔╝███████║\n\
"+headerLineOffset+" ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚══════╝\n"

    header += "\
"+programNameOffset + programName + "\n\
"+licenseOffset + licenseHeader + "\n\
"+configOffset + configHeader
    print(header)
    print(('#' * int(consoleColumns)))

#Determine console size if applicable
if systemOS != 'Windows':
    consoleRows, consoleColumns = os.popen('stty size', 'r').read().split()
elif systemOS == 'Windows':
    h = windll.kernel32.GetStdHandle(-12)
    csbi = create_string_buffer(22)
    res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
    (bufx, bufy, curx, cury, wattr, left, top, right, bottom, maxx, maxy) = struct.unpack("hhhhHhhhhhh", csbi.raw)
    consoleRows = bottom-top
    consoleColumns = right-left
        
def customTFBar_initialize_progbar(self, hook, epoch, logs=None):
        self.num_samples_seen = 0
        self.steps_to_update = 0
        self.steps_so_far = 0
        self.logs = defaultdict(float)
        self.num_epochs = self.params["epochs"]
        self.mode = "steps"
        self.total_steps = self.params["steps"]
        if hook == "train_overall":
            if self.show_overall_progress:
                self.overall_progress_tqdm = self.tqdm(
                    total=self.num_epochs,
                    bar_format=self.overall_bar_format,
                    leave=self.leave_overall_progress,
                    dynamic_ncols=True,
                    unit="epochs",
                    ascii=asciiFlag
                )
        elif hook == "test":
            if self.show_epoch_progress:
                self.epoch_progress_tqdm = self.tqdm(
                    total=self.total_steps,
                    desc="Evaluating",
                    bar_format=self.epoch_bar_format,
                    leave=self.leave_epoch_progress,
                    dynamic_ncols=True,
                    unit=self.mode,
                    ascii=asciiFlag
                )
        elif hook == "train_epoch":
            if self.show_epoch_progress:
                self.epoch_progress_tqdm = self.tqdm(
                    total=self.total_steps,
                    bar_format=self.epoch_bar_format,
                    leave=self.leave_epoch_progress,
                    dynamic_ncols=True,
                    unit=self.mode,
                    ascii=asciiFlag
                )

def customTFBar_on_epoch_end(self, epoch, logs={}):
    self._clean_up_progbar("train_epoch", logs)
    if self.show_overall_progress:
        metric_value_pairs = []
        for key, value in logs.items():
            if key in ["batch", "size"]: continue
            pair = self.metrics_format.format(name=key, value=value)
            metric_value_pairs.append(pair)
        metrics = self.metrics_separator.join(metric_value_pairs)
        self.overall_progress_tqdm.desc = metrics
        self.overall_progress_tqdm.update(1)

#Replace tqdm progress bar definitions from tensorflow-addons with customized versions
tfa.callbacks.TQDMProgressBar._initialize_progbar = customTFBar_initialize_progbar
tfa.callbacks.TQDMProgressBar.on_epoch_end = customTFBar_on_epoch_end

#Clear the screen and print out the program header
os.system('cls' if os.name=='nt' else 'clear')
programTitle(versionNum, configFileName)

