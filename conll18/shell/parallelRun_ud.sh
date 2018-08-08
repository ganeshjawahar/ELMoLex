
process(){
  eval $1
  exit 0
}

export -f process

#read commands
commands=()
while read line
do
  commands+=("$line")
done < $1

duration=$SECONDS
for command in "${commands[@]}";
do
  echo $command
done  | xargs -I {} --max-procs $2 bash -c 'process "$@"' _ {}

echo "Exit code for xargs = $?"
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds took to complete."