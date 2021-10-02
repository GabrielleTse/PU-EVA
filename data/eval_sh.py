import glob

off_files = glob.glob("test/*.off")

f = open("eval.sh",'w')
for num in [1024,8192,16384]:
    
    for off_file in off_files:
        xyz_file = "pred_xyz/{}/{}".format(num,off_file.split("/")[-1].replace(".off",".xyz"))
        print(num,xyz_file)
        f.write("./evaluation {} {}\n".format(off_file,xyz_file))
        

