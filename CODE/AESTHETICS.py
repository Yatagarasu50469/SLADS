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
    elif 'DLADS' in erdModel: 
        programName = "Deep Learning Approach for Dynamic Sampling"
        headerWidth = 40
    elif erdModel == 'GLANDS': 
        programName = "Generative Learning Adversarial Networks for Dynamic Sampling"
        headerWidth = 50
    else:
        sys.exit('\nError - Unknown erdModel specified.')
    programName += " - v"+versionNum
    programNameOffset = (' ' * int((int(consoleColumns)-len(programName))//2))
    headerLineOffset = (' ' * int((int(consoleColumns)-headerWidth)//2))
    licenseOffset = (' ' * int((int(consoleColumns)-len(licenseHeader))//2))
    configOffset = (' ' * int((int(consoleColumns)-len(configHeader))//2))
    
    #Font: textfancy - Dark with Shadow
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

    elif 'DLADS' in erdModel: 
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

#Clear the screen and print out the program header
os.system('cls' if os.name=='nt' else 'clear')
programTitle(versionNum, configFileName)

