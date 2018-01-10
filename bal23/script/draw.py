WIDTH = 221
HEIGHT = 100
MID = int(WIDTH/2)


def expand_cigar(cigar):
    num = 0
    ret = ""
    for c in cigar:
        if c.isdigit():
            num = num * 10 + int(c)
        else:
            ret = ret + c * num
            num = 0
    return ret

def base_qual_poor(qual, cigar, pos):
    cigar_idx = 0
    seq_idx = 0
    mypos = 0
    while seq_idx < len(qual):
        this_cigar = cigar[cigar_idx]
        cigar_idx = cigar_idx + 1
        if this_cigar == 'M':
            if mypos == pos: return seq_idx
            mypos = mypos+1
            seq_idx = seq_idx+1 
        elif this_cigar == 'I' or this_cigar == 'S':
            seq_idx = seq_idx+1
        elif this_cigar == 'D':
            if mypos == pos: return -1 #this is the case when there isn't a base actually, so return 0 temporarily
            mypos = mypos+1
        elif this_cigar == 'H':
            pass
        else:
            raise Exception

def main():
    import sys
    if len(sys.argv) <= 1:
        #print "Usage %s <filename prefix> <pos>" % (sys.argv[0])
        exit()
    filename = sys.argv[1]
    #with open(filename+".fa") as f:
        #ref = f.read().translate(None,"\n")
    with open(filename+".sam") as f:
        sam = f.read().splitlines()

    call_pos = int(sys.argv[2])
    base_12 = [0]  * 12
    
    weight = dict.fromkeys(range(5,HEIGHT))
    #print("A	C	G	T	seq_idx	mapq	qual	stra")
    for sam_line in sam:
        line = sam_line.split("\t")
        pos = int(line[3])
        mapq = int(line[4])
        seq = line[9]
        qual = line[10]
        flag = int(line[1])
        revcomp = bool(flag & 16)
        fail = bool((flag & 256) or (flag & 2048) or (flag & 2 == 0) or (flag & 512) or (flag & 4) or (flag & 1024)) 
        if fail: 
	    #print("Here fails")
            continue
        cigar = expand_cigar(line[5])
	seq_pos = base_qual_poor(qual, cigar, call_pos - pos)

	if mapq <= 1: continue
	#quality score
	if seq_pos == -1: #return -1 means it is deleted, so set quality score to zero
	    quality_score = 0
	else:
            quality_score = ord(qual[seq_pos])
	
        #base_idx
        if seq_pos == -1:
	    #print("there is a break")
	    continue
        else:
            base = seq[seq_pos]
            if base == "A":
                base_idx = 0
            elif base == "C":
                base_idx = 1
            elif base == "G":
                base_idx = 2
            elif base == "T":
                base_idx = 3

	if quality_score <= 10:
	    base_12[3 * base_idx] += 0
        elif quality_score <=13:
	    base_12[3 * base_idx] += 1
	elif quality_score <=17:
            base_12[3 * base_idx] += 2
	elif quality_score <=20:
            base_12[3 * base_idx] += 3
	else:
            base_12[3 * base_idx] += 4


	#base
	base_four = [0] * 4
	base_four[base_idx] = 1


	#strand direction
	if revcomp:
	    strand_direction = 0
	    base_12[3 * base_idx + 2] += 1
	else:
	    strand_direction = 1
	    base_12[3 * base_idx + 1] += 1

	# check base qual at call position
        #if base_qual_poor(qual, cigar, call_pos - pos): continue;
        #print("This is mapq %s, this is quality score %s, this is strand direction %s, this is base %s%s%s%s" % (mapq, quality_score, strand_direction, base_four[0], base_four[1], base_four[2], base_four[3]))
	print("%s	%s	%s	%s	%s	%s	%s	%s" % (base_four[0], base_four[1], base_four[2], base_four[3], seq_pos, mapq, quality_score, strand_direction))
    #base_arr = string.join(base_12, "	")
    base_arr = "	".join(str(x) for x in base_12)
    #print(base_arr)
if __name__ == "__main__":
    main()
