a=1
for i in *.png; do
	new=$(printf "zxc_%d.png" "$a") #04 pad to length of 4
  	mv -i -- "$i" "$new"
  	let a=a+1
done
a=1
for i in *.png; do
	new=$(printf "t_%d.png" "$a") #04 pad to length of 4
  	mv -i -- "$i" "$new"
  	let a=a+1
done
