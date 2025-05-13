import os
import subprocess
import sys

def process_n4_correction(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "T1" in os.path.basename(dirpath):
            for filename in filenames:
                if filename.endswith(".nii.gz"):
                    input_path = os.path.join(dirpath, filename)
                    base = filename[:-7]
                    corrected_name = "corrected_%s.nii.gz" % base
                    bias_name = "bias_field.%s.nii.gz" % base
                    
                    corrected_path = os.path.join(dirpath, corrected_name)
                    bias_path = os.path.join(dirpath, bias_name)

                    cmd = [
                        "N4BiasFieldCorrection",
                        "-d", "3",
                        "-v", "1",
                        "-s", "4",
                        "-b", "[120]",
                        "-c", "[80x80x80x80 ,0.0005]",
                        "-i", input_path,
                        "--histogram-sharpening ","[0.15,0.01,200]",
                        "-o", "[%s,%s]" % (corrected_path, bias_path)
                    ]

                    print "\nProcessing:", input_path
                    try:
                        subprocess.check_call(cmd)
                        print "Saved to:", corrected_path
                        print "Bias field saved to:", bias_path
                    except subprocess.CalledProcessError as e:
                        print "Process Error:", str(e)
                    except Exception as e:
                        print "Unexpected Error:", str(e)

if __name__ == "__main__":
    target_dir = "/home/yangyucheng/test/1.5T"
    if not os.path.exists(target_dir):
        print "Error: Directory %s not found!" % target_dir
        sys.exit(1)
    process_n4_correction(target_dir)