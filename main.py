import csv
import datetime

CSV_FNAME1 = "PPMI_Curated_Data_Cut_Public_20230612.csv"
#CSV_FNAME1 = "small1.csv"
CSV_FNAME2 = "Blood_Chemistry___Hematology_25Jul2023.csv"
#CSV_FNAME2 = "small.csv"
CSV_FNAME3 = "Olfactory_UPSIT-Archived_25Jul2023.csv"

CSV_OUTPUT = 'work.csv'
FIELD_SEPARATOR = ','

# INPUT CODES 1
FIELD1_N_PATNO = 1
FIELD1_N_SUBGROUP = 4
FIELD1_N_EVENTID = 14
FIELD1_N_AGE_VISIT = 18
FIELD1_N_SEX = 19
FIELD1_N_EDUCYRS = 20
FIELD1_N_RACE = 22
FIELD1_N_FAMPD = 27
FIELD1_N_HANDED = 29
FIELD1_N_BMI = 32
FIELD1_N_AGEDIAG = 33
FIELD1_N_AGEONSET = 34
FIELD1_N_DURATION = 35
FIELD1_N_UPSITPCTL = 50
FIELD1_N_MOCA = 51
FIELD1_N_BJLOT = 52
FIELD1_N_HVLT_DISCRIMINATION = 54
FIELD1_N_SDMTOTAL = 63
FIELD1_N_VLTANIM = 66
FIELD1_N_MSEADLG = 69
FIELD1_N_ESS = 79
FIELD1_N_REM = 80
FIELD1_N_STAI = 82
FIELD1_N_SCOPA = 85
FIELD1_N_NHY = 96
FIELD1_N_UPDRS_TOT = 114
FIELD1_N_TAU = 127

# INPUT CODES 2
FIELD2_N_PATNO = 0
FIELD2_N_EVENTID = 1
FIELD2_N_LISTCODE = 11
FIELD2_N_LSIRES = 14

# INPUT CODES 3
FIELD3_N_PATNO = 0
FIELD3_N_UPSIIT_PRCNTG = 85

def add_virgolette(s):
    return '\"' + s + '\"'


def write_header(f):
    print(add_virgolette("PATNO"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("subgroup"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("age_at_visit"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("sex"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("educyrs"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("race"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("fampd"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("handed"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("BMI"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("agediag"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("ageonset"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("duration"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("upsitpctl"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("MOCA"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("bjlot"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("hvlt_discrimination"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("SMD_total"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("vltanim"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("mseadlg"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("ess"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("rem"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("STAI"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("SCOPA"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("NHY"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("UPDRS_tot"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("TAU"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("PATNO2"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("RCT4"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("CGT284"), file=f, end=FIELD_SEPARATOR)
    print(add_virgolette("RCT5"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("RCT13"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("RCT1487"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("HMT12"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("HMT19"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("RCT183"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("RCT392"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("HMT11"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("HMT18"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("HMT2"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("HMT40"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("HMT9"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("HMT16"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("HMT10"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("HMT17"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("HMT8"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("HMT15"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("HMT13"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("CGT283"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("HMT3"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("RCT17"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("RCT18"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("RCT11"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("RCT16"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("RCT15"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("RCT8"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("RCT1"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("RCT12"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("RCT6"), file=f,end=FIELD_SEPARATOR)
    print(add_virgolette("HMT7"), file=f)
    return 0


def create_csv():
    csv_file1 = open(CSV_FNAME1, 'r')
    #csv_file2 = open(CSV_FNAME2, 'r')

    foutput = open(CSV_OUTPUT, 'w')
    csv_reader1 = csv.reader(csv_file1, delimiter=FIELD_SEPARATOR)
    #csv_reader2 = csv.reader(csv_file2, delimiter=FIELD_SEPARATOR)
    line_count = 0
    header = next(csv_reader1)  # store the headers and advance reader pointer
    write_header(foutput)
    start = datetime.datetime.now().replace(microsecond=0)
    for row in csv_reader1:
        #################################
        # EVENT_ID
        #################################
        event_id = row[FIELD1_N_EVENTID]
        if event_id != "BL":
            continue

        #################################
        # PATNO
        #################################
        patno = row[FIELD1_N_PATNO]
        print(patno)

        #################################
        # other fields
        #################################
        subgroup = row[FIELD1_N_SUBGROUP]
        age_at_visit = float(row[FIELD1_N_AGE_VISIT])
        if age_at_visit > 1000:
            age_at_visit = age_at_visit / 1000.0
        age_at_visit = str(age_at_visit)
        sex = row[FIELD1_N_SEX]
        educyrs = row[FIELD1_N_EDUCYRS]
        race = row[FIELD1_N_RACE]
        fampd = row[FIELD1_N_FAMPD]
        handed = row[FIELD1_N_HANDED]
        bmi = row[FIELD1_N_BMI]
        agediag = row[FIELD1_N_AGEDIAG]
        ageonset = row[FIELD1_N_AGEONSET]
        duration = row[FIELD1_N_DURATION]
        upsitpctl = row[FIELD1_N_UPSITPCTL]
        moca = row[FIELD1_N_MOCA]
        bjlot = row[FIELD1_N_BJLOT]
        hvlt_discrimination = row[FIELD1_N_HVLT_DISCRIMINATION]
        smd_total = row[FIELD1_N_SDMTOTAL]
        vltanim = row[FIELD1_N_VLTANIM]
        mseadlg = row[FIELD1_N_MSEADLG]
        ess = row[FIELD1_N_ESS]
        rem = row[FIELD1_N_REM]
        stai = row[FIELD1_N_STAI]
        scopa = row[FIELD1_N_SCOPA]
        nhy = row[FIELD1_N_NHY]
        updrs_tot = row[FIELD1_N_UPDRS_TOT]
        tau = row[FIELD1_N_TAU]

        #################################
        # SECOND FILE
        #################################
        csv_file2 = open(CSV_FNAME2, 'r')
        csv_reader2 = csv.reader(csv_file2, delimiter=FIELD_SEPARATOR)
        header = next(csv_reader2)  # store the headers and advance reader pointer
        rct4 = ""
        cgt284 = ""
        rct5 = ""
        rct13 = ""
        rct1407 = ""
        hmt12 = ""
        hmt19 = ""
        rct183 = ""
        rct392 = ""
        hmt11 = ""
        hmt18 = ""
        hmt2 = ""
        hmt40 = ""
        hmt9 = ""
        hmt16 = ""
        hmt10 = ""
        hmt17 = ""
        hmt8 = ""
        hmt15 = ""
        hmt13 = ""
        cgt283 = ""
        hmt3 = ""
        hmt71 = ""
        rct17 = ""
        rct18 = ""
        rct11 = ""
        rct16 = ""
        rct15 = ""
        rct8 = ""
        rct1 = ""
        rct12 = ""
        rct6 = ""
        hmt7 = ""

        found = False
        last_pat = -1
        for row2 in csv_reader2:
            patno2 = row2[FIELD2_N_PATNO]
            event_id2 = row2[FIELD2_N_EVENTID]
            if event_id2 != "SC":
                continue
            if patno != patno2:
                if found:
                    # exit from inner loop
                    break
                else:
                    continue
            else:
                found = True
                last_pat = patno2

            code = row2[FIELD2_N_LISTCODE]
            value = row2[FIELD2_N_LSIRES]

            if code == "RCT4":
                rct4 = value
            elif code == "CGT284":
                cgt284 = value
            elif code == "CGT626":
                cgt626 = value
            elif code == "RCT5":
                rct5 = value
            elif code == "RCT13":
                rct13 = value
            elif code == "RCT1407":
                rct1407 = value
            elif code == "HMT12":
                hmt12 = value
            elif code == "HMT19":
                hmt19 = value
            elif code == "RCT183":
                rct183 = value
            elif code == "RCT392":
                rct392 = value
            elif code == "HMT11":
                hmt11 = value
            elif code == "HMT18":
                hmt18 = value
            elif code == "HMT2":
                hmt2 = value
            elif code == "HMT40":
                hmt40 = value
            elif code == "HMT9":
                hmt9 = value
            elif code == "HMT16":
                hmt16 = value
            elif code == "HMT10":
                hmt10 = value
            elif code == "HMT17":
                hmt17 = value
            elif code == "HMT8":
                hmt8 = value
            elif code == "HMT15":
                hmt15 = value
            elif code == "HMT13":
                hmt13 = value
            elif code == "CGT283":
                cgt283 = value
            elif code == "HMT3":
                hmt3 = value
            elif code == "HMT71":
                hmt71 = value
            elif code == "RCT17":
                rct17 = value
            elif code == "RCT18":
                rct18 = value
            elif code == "RCT11":
                rct11 = value
            elif code == "RCT16":
                rct16 = value
            elif code == "RCT15":
                rct15 = value
            elif code == "RCT8":
                rct8 = value
            elif code == "RCT1":
                rct1 = value
            elif code == "RCT12":
                rct12 = value
            elif code == "RCT6":
                rct6 = value
            elif code == "HMT7":
                hmt7 = value
            else:
                continue

        if last_pat == -1:
            continue

        #################################
        # OUTPUT record
        #################################
        # print first file fields
        print(patno, end=FIELD_SEPARATOR, file=foutput)
        print(subgroup, end=FIELD_SEPARATOR, file=foutput)
        print(age_at_visit, end=FIELD_SEPARATOR, file=foutput)
        print(sex, end=FIELD_SEPARATOR, file=foutput)
        print(educyrs, end=FIELD_SEPARATOR, file=foutput)
        print(race, end=FIELD_SEPARATOR, file=foutput)
        print(fampd, end=FIELD_SEPARATOR, file=foutput)
        print(handed, end=FIELD_SEPARATOR, file=foutput)
        print(bmi, end=FIELD_SEPARATOR, file=foutput)
        print(agediag, end=FIELD_SEPARATOR, file=foutput)
        print(ageonset, end=FIELD_SEPARATOR, file=foutput)
        print(duration, end=FIELD_SEPARATOR, file=foutput)
        print(upsitpctl, end=FIELD_SEPARATOR, file=foutput)
        print(moca, end=FIELD_SEPARATOR, file=foutput)
        print(bjlot, end=FIELD_SEPARATOR, file=foutput)
        print(hvlt_discrimination, end=FIELD_SEPARATOR, file=foutput)
        print(smd_total, end=FIELD_SEPARATOR, file=foutput)
        print(vltanim, end=FIELD_SEPARATOR, file=foutput)
        print(mseadlg, end=FIELD_SEPARATOR, file=foutput)
        print(ess, end=FIELD_SEPARATOR, file=foutput)
        print(rem, end=FIELD_SEPARATOR, file=foutput)
        print(stai, end=FIELD_SEPARATOR, file=foutput)
        print(scopa, end=FIELD_SEPARATOR, file=foutput)
        print(nhy, end=FIELD_SEPARATOR, file=foutput)
        print(updrs_tot, end=FIELD_SEPARATOR, file=foutput)
        print(tau, end=FIELD_SEPARATOR, file=foutput)


        #################################
        # OUTPUT record second file
        #################################
        # print record
        print(last_pat, end=FIELD_SEPARATOR, file=foutput)
        print(rct4, end=FIELD_SEPARATOR, file=foutput)
        print(cgt284, end=FIELD_SEPARATOR, file=foutput)
        print(rct5, end=FIELD_SEPARATOR, file=foutput)
        print(rct13, end=FIELD_SEPARATOR, file=foutput)
        print(rct1407, end=FIELD_SEPARATOR, file=foutput)
        print(hmt12, end=FIELD_SEPARATOR, file=foutput)
        print(hmt19, end=FIELD_SEPARATOR, file=foutput)
        print(rct183, end=FIELD_SEPARATOR, file=foutput)
        print(rct392, end=FIELD_SEPARATOR, file=foutput)
        print(hmt11, end=FIELD_SEPARATOR, file=foutput)
        print(hmt18, end=FIELD_SEPARATOR, file=foutput)
        print(hmt2, end=FIELD_SEPARATOR, file=foutput)
        print(hmt40, end=FIELD_SEPARATOR, file=foutput)
        print(hmt9, end=FIELD_SEPARATOR, file=foutput)
        print(hmt16, end=FIELD_SEPARATOR, file=foutput)
        print(hmt10, end=FIELD_SEPARATOR, file=foutput)
        print(hmt17, end=FIELD_SEPARATOR, file=foutput)
        print(hmt8, end=FIELD_SEPARATOR, file=foutput)
        print(hmt15, end=FIELD_SEPARATOR, file=foutput)
        print(hmt13, end=FIELD_SEPARATOR, file=foutput)
        print(cgt283, end=FIELD_SEPARATOR, file=foutput)
        print(hmt3, end=FIELD_SEPARATOR, file=foutput)
        print(rct17, end=FIELD_SEPARATOR, file=foutput)
        print(rct18, end=FIELD_SEPARATOR, file=foutput)
        print(rct11, end=FIELD_SEPARATOR, file=foutput)
        print(rct16, end=FIELD_SEPARATOR, file=foutput)
        print(rct15, end=FIELD_SEPARATOR, file=foutput)
        print(rct8, end=FIELD_SEPARATOR, file=foutput)
        print(rct1, end=FIELD_SEPARATOR, file=foutput)
        print(rct12, end=FIELD_SEPARATOR, file=foutput)
        print(rct6, end=FIELD_SEPARATOR, file=foutput)
        print(hmt7, file=foutput)

        # #################################
        # # THIRD FILE
        # #################################
        # csv_file3 = open(CSV_FNAME3, 'r')
        # csv_reader3 = csv.reader(csv_file3, delimiter=FIELD_SEPARATOR)
        # header = next(csv_reader3)  # store the headers and advance reader pointer
        # found = False
        # upsiit_percentage = ""
        # patno3 = ""
        # for row3 in csv_reader3:
        #     patno3 = row3[FIELD3_N_PATNO]
        #
        #     if patno == patno3:
        #         found = True
        #         upsiit_percentage = row3[FIELD3_N_UPSIIT_PRCNTG]
        #         break
        #
        # #################################
        # # OUTPUT record third file
        # #################################
        # # print record
        # print(patno3, end=FIELD_SEPARATOR, file=foutput)
        # print(upsiit_percentage, file=foutput)

        line_count += 1
        # print(".", end="")
        # if line_count % 100 == 0:
        #     print("\n")

    print(f'Found {line_count} participants.')
    end = datetime.datetime.now().replace(microsecond=0)
    run_time = end - start
    print("Running time: {}\n".format(run_time))



def main():
    create_csv()



if __name__ == '__main__':
    main()


