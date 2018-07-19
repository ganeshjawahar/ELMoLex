import glob
import os
import sys

CL_HOME=os.environ["CL_HOME"]
#model_folder = CL_HOME+"/system1"
#model_folder = "/home/ganesh/objects/finale/june16/direct"
model_folder = CL_HOME+"/"+sys.argv[1]
#upload_folder = CL_HOME+"/upload_june29"
upload_folder = CL_HOME + "/upload_"+sys.argv[1]
command_txt = "tmp/shell/upload.sh"

os.makedirs(upload_folder)
w_up = open(command_txt, 'w')
for src_folder in glob.glob(model_folder+"/*"):
  #src_folder = "/home/ganesh/objects/neurogp/en/test_lex_expand"
  tb_name = src_folder.split("/")[-1]
  cmds = []
  cmds.append("mkdir "+upload_folder+"/"+tb_name)
  cur_tb_dest = upload_folder+"/"+tb_name
  for sub_src_folder in glob.glob(src_folder+"/*"):
    if not os.path.isdir(sub_src_folder):
      # its a file
      cmds.append("cp "+sub_src_folder+" "+cur_tb_dest)
    else:
      # its a folder
      sub_src_folder_name = sub_src_folder.split("/")[-1]
      if sub_src_folder_name.startswith("fold"):
        cmds.append("mkdir "+cur_tb_dest+"/"+sub_src_folder_name)
        cur_src_sub_folder = sub_src_folder
        cur_dest_sub_folder = cur_tb_dest+"/"+sub_src_folder_name
        for sub_folder in glob.glob(cur_src_sub_folder+"/*"):
          if not os.path.isdir(sub_folder):
            # its a file
            cmds.append("cp "+sub_folder+" "+cur_dest_sub_folder)
          else:
            # its a folder
            folder_name = sub_folder.split("/")[-1]
            if folder_name!="model":
              cmds.append("cp -r "+sub_folder+" "+cur_dest_sub_folder)
            else:
              #cmds.append("mkdir "+cur_dest_sub_folder)
              # find best model
              model_id = [ int(file.split('/')[-1].split('_')[-1].split('.')[0]) for file in glob.glob(os.path.join(sub_folder, 'model_*'))]
              if len(model_id)>0:
                model_id = max(model_id)
                model_id = str(model_id)
                model_abs_path = os.path.join(sub_folder, 'model_epoch_'+str(model_id)+'.pt')
                cmds.append("cp "+model_abs_path+" "+cur_dest_sub_folder)
              # find non-model files (if any)
              for file in glob.glob(os.path.join(sub_folder, '*')):
                if not file.split("/")[-1].startswith("model_"):
                  cmds.append("cp "+file+" "+upload_folder+"/"+cur_dest_sub_folder)
      elif sub_src_folder_name not in ["elmo", "model", "ltrans", "nlm", "lm"]:
        cmds.append("cp -r "+sub_src_folder+" "+cur_tb_dest)
      else:
        cmds.append("mkdir "+cur_tb_dest+"/"+sub_src_folder_name)
        # find best model
        model_id = [ int(file.split('/')[-1].split('_')[-1].split('.')[0]) for file in glob.glob(os.path.join(sub_src_folder, 'model_*'))]
        if len(model_id)>0:
          model_id = max(model_id)
          model_id = str(model_id)
          model_abs_path = os.path.join(sub_src_folder, 'model_epoch_'+str(model_id)+'.pt')
          cmds.append("cp "+model_abs_path+" "+cur_tb_dest+"/"+sub_src_folder_name)
        # find non-model files (if any)
        for file in glob.glob(os.path.join(sub_src_folder, '*')):
          if not file.split("/")[-1].startswith("model_"):
            cmds.append("cp "+file+" "+upload_folder+"/"+sub_src_folder_name)
  for cmd in cmds:
    w_up.write(cmd+"\n")
w_up.close()



