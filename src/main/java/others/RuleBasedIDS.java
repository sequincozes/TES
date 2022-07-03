package others;

import avaliacao_pratica.Auxiliar;
import weka.core.Instance;
import weka.core.Instances;

import java.io.IOException;

public class RuleBasedIDS {

    static double maxTime = 1000;
    static Instances testingDataset;

    public static void main(String[] args) throws IOException {
//        testingDataset = Auxiliar.lerDataset("/home/silvio/datasets/ereno_goose65_test.arff");
        testingDataset = Auxiliar.lerDataset("/home/silvio/datasets/ereno_goose_test.arff");

        System.out.println("R1,R2,R3,R4,R5,R6,R7,R8,R9");
        for (int i = 1; i < testingDataset.size() - 1; i++) {
            System.out.println("" + checkR1(testingDataset.get(i)) + ","+
                    checkR2(testingDataset.get(i)) + ", " +
                    checkR3(testingDataset.get(i)) + ", " +
                    checkR4(testingDataset.get(i)) + ", " +
                    checkR5(testingDataset.get(i)) + ", " +
                    checkR6(testingDataset.get(i), testingDataset.get(i - 1)) + ", " +
                    checkR7(testingDataset.get(i)) + ", " +
                    checkR8(testingDataset.get(i)) + ", " +
                    checkR9(testingDataset.get(i)));

        }

    }

    //R1	GOOSE messages must have MAC address starting with 01-0c-cd-01
    private static boolean checkR1(Instance instance) {
        String ethDest = String.valueOf(instance.value(6));
        String ethSrc = String.valueOf(instance.value(7));
        if (!ethDest.startsWith("01-0c-cd-01") && ethSrc.startsWith("01-0c-cd-01")) {
            return true; // it is an attack
        }
        return false;
    }

    //R2	GOOSE messages must have the TPID field with value 0x8100
    private static boolean checkR2(Instance instance) {
        String TPID = String.valueOf(instance.value(12));
        if (!TPID.equals("0x8100")) {
            return true; // it is an attack
        }
        return false;
    }

    //R3	GOOSE messages must have the ethertype field equal to 0x88B8
    private static boolean checkR3(Instance instance) {
        String ethertype = String.valueOf(instance.value(8));
        if (!ethertype.equals("0x000088b8")) {
            return true; // it is an attack
        }
        return false;
    }

    //R4	GOOSE messages must have TimeAllowedToLive equal to double of the value of MaxTime
    private static boolean checkR4(Instance instance) {
        double gooseTimeAllowedtoLive = instance.value(9);
        if (gooseTimeAllowedtoLive != (2 * maxTime)) {
            return true; // it is an attack
        }
        return false;
    }

    //R5	GOOSE messages must have the APPID field formatted as a 4-byte hex-decimal (e.g, 0000-3FFF)
    private static boolean checkR5(Instance instance) {
        String gooseAppidHex = String.valueOf(instance.stringValue(10));
        if (gooseAppidHex.startsWith("0x") & gooseAppidHex.substring(2).length() == 8) {
            try {
                Integer.parseInt(gooseAppidHex.substring(2));
            } catch (Exception e) {
                return true;
            }
        } else {
            return true;
        }
        return false;
    }

    //R6	Consecutive GOOSE messages must have consistent values for fields gocbRef, timeAllowedToLive, datSet, goID, t, StNum, SqNum, test, confRev, ndsCom and numDatSetEntries
    private static boolean checkR6(Instance gooseMessage, Instance previousMessage) {
        boolean debug = false;

        // Equals consistency
        boolean gocbRef = String.valueOf(gooseMessage.value(13)).equals(String.valueOf(previousMessage.value(13)));
        boolean timeAllowedToLive = String.valueOf(gooseMessage.value(9)).equals(String.valueOf(previousMessage.value(9)));
        boolean datSet = String.valueOf(gooseMessage.value(14)).equals(String.valueOf(previousMessage.value(14)));
        boolean goID = String.valueOf(gooseMessage.value(15)).equals(String.valueOf(previousMessage.value(15)));
        boolean test = String.valueOf(gooseMessage.value(16)).equals(String.valueOf(previousMessage.value(16)));
        boolean confRev = String.valueOf(gooseMessage.value(17)).equals(String.valueOf(previousMessage.value(17)));
        boolean ndsCom = String.valueOf(gooseMessage.value(18)).equals(String.valueOf(previousMessage.value(18)));
        boolean numDatSetEntries = String.valueOf(gooseMessage.value(19)).equals(String.valueOf(previousMessage.value(19)));

        if (!(gocbRef & timeAllowedToLive & datSet & goID & test & confRev & ndsCom & numDatSetEntries)) {
            if (debug) {
                System.out.println("Some field is not consistent");
            }
            return true; // it is an attack
        }

        // Sequential consistency
        if (!(gooseMessage.value(28) > 0)) { //timestampDiff should be > 0
            if (debug) {
                System.out.println("timestampDiff should be > 0!");
            }
            return true; // it is an attack
        }
        if ((gooseMessage.value(22) > 1)) { // stDiff should not be > 1
            if (debug) {
                System.out.println("// stDiff should not be > 1");
            }
            return true; // it is an attack
        }

        if ((gooseMessage.value(22) == 1) & (gooseMessage.value(23) != 0)) { // stDiff is = 1 and sqDiff != 0 OR
            // not a problem
        } else if (gooseMessage.value(22) == 0 & gooseMessage.value(23) == 1) {// stDiff is = 0, sqDiff == 1
            // not a problem
        } else {
            if (debug) {
                System.out.println("This is not true: stDiff is = 1 and sqDiff != 0 OR stDiff is = 0, sqDiff == 1");
            }
            return true;
        }

        return false;
    }

    //R7	GOOSE messages must have the APPID field matching the last two octets of the destination multicast address
    private static boolean checkR7(Instance instance) {
        String ethDest = String.valueOf(instance.value(6));
        String gooseAppidHex = String.valueOf(instance.stringValue(10));
        if (!ethDest.endsWith(gooseAppidHex)) {
            return true; // it is an attack
        }
        return false;
    }

    //R8	The IED control block name must be consistent with the value of the goID field
    // (i.e., the 𝐿𝐷/𝐿𝑁 value in the gocoRef field must match the datSet field from the GOOSE APDU)
    private static boolean checkR8(Instance instance) {
        String goID = String.valueOf(instance.value(15)).replace("Interlocking", "IntLock"); // Corrige abreviação do ERENO
        String datSet = String.valueOf(instance.value(14));
        if (!datSet.contains(goID)) {
            return true; // it is an attack
        }
        return false;
    }

    //R9	The size of frames containing GOOSE messages should be equal to 8 𝑏𝑦𝑡𝑒𝑠+ 𝐴𝑃𝐷𝑈 𝑠𝑖𝑧𝑒, and 𝐴𝑃𝐷𝑈 𝑠𝑖𝑧𝑒 should be less than 1492 𝑏𝑦𝑡𝑒𝑠
    // TODO DAQUI PRA BAIXO
    private static boolean checkR9(Instance instance) {
        String ethDest = String.valueOf(instance.value(6));
        String ethSrc = String.valueOf(instance.value(7));
        if (!ethDest.startsWith("01-0c-cd-01") && ethSrc.startsWith("01-0c-cd-01")) {
            return true; // it is an attack
        }
        return false;
    }

    //R10	The SqNum in GOOSE messages should be set to zero whenever the value of the StNum changes (w.r.t the previous message)
    private static boolean checkR10(Instance instance) {
        String ethDest = String.valueOf(instance.value(6));
        String ethSrc = String.valueOf(instance.value(7));
        if (!ethDest.startsWith("01-0c-cd-01") && ethSrc.startsWith("01-0c-cd-01")) {
            return true; // it is an attack
        }
        return false;
    }

    //R11	The number of messages captured in an interval must not exceed a pre-defined threshold (20% above the expected maximum)
    private static boolean checkR11(Instance instance) {
        String ethDest = String.valueOf(instance.value(6));
        String ethSrc = String.valueOf(instance.value(7));
        if (!ethDest.startsWith("01-0c-cd-01") && ethSrc.startsWith("01-0c-cd-01")) {
            return true; // it is an attack
        }
        return false;
    }

    //R12	The number of messages captured in an interval must not be equal to zero
    private static boolean checkR12(Instance instance) {
        String ethDest = String.valueOf(instance.value(6));
        String ethSrc = String.valueOf(instance.value(7));
        if (!ethDest.startsWith("01-0c-cd-01") && ethSrc.startsWith("01-0c-cd-01")) {
            return true; // it is an attack
        }
        return false;
    }

    //R13	The transmitter’s timestamp should not be higher than the receiver’s timestamp
    private static boolean checkR13(Instance instance) {
        String ethDest = String.valueOf(instance.value(6));
        String ethSrc = String.valueOf(instance.value(7));
        if (!ethDest.startsWith("01-0c-cd-01") && ethSrc.startsWith("01-0c-cd-01")) {
            return true; // it is an attack
        }
        return false;
    }

    //R14	The transmitter’s timestamp from GOOSE messages should not be more than 4 𝑚𝑠 apart from the receiver’s timestamp
    private static boolean checkR14(Instance instance) {
        String ethDest = String.valueOf(instance.value(6));
        String ethSrc = String.valueOf(instance.value(7));
        if (!ethDest.startsWith("01-0c-cd-01") && ethSrc.startsWith("01-0c-cd-01")) {
            return true; // it is an attack
        }
        return false;
    }

    //R15	The Recency metric, represented by the last GOOSE message’s arrival, must respect a minimum and a maximum threshold
    private static boolean checkR15(Instance instance) {
        String ethDest = String.valueOf(instance.value(6));
        String ethSrc = String.valueOf(instance.value(7));
        if (!ethDest.startsWith("01-0c-cd-01") && ethSrc.startsWith("01-0c-cd-01")) {
            return true; // it is an attack
        }
        return false;
    }

    //R16	The Frequency metric, represented by the average number of received GOOSE messages, must respect a minimum and a maximum predefined threshold
    private static boolean checkR16(Instance instance) {
        String ethDest = String.valueOf(instance.value(6));
        String ethSrc = String.valueOf(instance.value(7));
        if (!ethDest.startsWith("01-0c-cd-01") && ethSrc.startsWith("01-0c-cd-01")) {
            return true; // it is an attack
        }
        return false;
    }

    //R17	The Monetary metric, represented by the total number of received GOOSE messages, must be within a predefined threshold [46]. The difference from rule #𝑅11) is that this rule considers only received GOOSE messages
    private static boolean checkR17(Instance instance) {
        String ethDest = String.valueOf(instance.value(6));
        String ethSrc = String.valueOf(instance.value(7));
        if (!ethDest.startsWith("01-0c-cd-01") && ethSrc.startsWith("01-0c-cd-01")) {
            return true; // it is an attack
        }
        return false;
    }

    //R18	Only messages with specific source port, IP and MAC addresses are allowed
    private static boolean checkR18(Instance instance) {
        String ethDest = String.valueOf(instance.value(6));
        String ethSrc = String.valueOf(instance.value(7));
        if (!ethDest.startsWith("01-0c-cd-01") && ethSrc.startsWith("01-0c-cd-01")) {
            return true; // it is an attack
        }
        return false;
    }

    //R19	Only MMS, COTP, TPKT, and SNTP protocols are allowed on the station level network and only the GOOSE, SV, and IEEE 1588 protocols are allowed on the process level network
    private static boolean checkR19(Instance instance) {
        String ethDest = String.valueOf(instance.value(6));
        String ethSrc = String.valueOf(instance.value(7));
        if (!ethDest.startsWith("01-0c-cd-01") && ethSrc.startsWith("01-0c-cd-01")) {
            return true; // it is an attack
        }
        return false;
    }

    //R20	There must be consistency between the GOOSE switch-in messages (e.g., breaker opening) and the value of the report sent by the MMS protocol (i.e., MMS signal report)
    private static boolean checkR20(Instance instance) {
        String ethDest = String.valueOf(instance.value(6));
        String ethSrc = String.valueOf(instance.value(7));
        if (!ethDest.startsWith("01-0c-cd-01") && ethSrc.startsWith("01-0c-cd-01")) {
            return true; // it is an attack
        }
        return false;
    }

    //R21	The number of bytes that travel per second must not exceed a predefined threshold
    private static boolean checkR21(Instance instance) {
        String ethDest = String.valueOf(instance.value(6));
        String ethSrc = String.valueOf(instance.value(7));
        if (!ethDest.startsWith("01-0c-cd-01") && ethSrc.startsWith("01-0c-cd-01")) {
            return true; // it is an attack
        }
        return false;
    }

    //R22	The number of packets that travel per second must not exceed a predefined threshold
    private static boolean checkR22(Instance instance) {
        String ethDest = String.valueOf(instance.value(6));
        String ethSrc = String.valueOf(instance.value(7));
        if (!ethDest.startsWith("01-0c-cd-01") && ethSrc.startsWith("01-0c-cd-01")) {
            return true; // it is an attack
        }
        return false;
    }

    //R23	The length of the packet (specified in the packet header) must not exceed a predefined threshold
    private static boolean checkR23(Instance instance) {
        String ethDest = String.valueOf(instance.value(6));
        String ethSrc = String.valueOf(instance.value(7));
        if (!ethDest.startsWith("01-0c-cd-01") && ethSrc.startsWith("01-0c-cd-01")) {
            return true; // it is an attack
        }
        return false;
    }

    //R24	The total size of the packet must not exceed a predefined threshold
    private static boolean checkR24(Instance instance) {
        String ethDest = String.valueOf(instance.value(6));
        String ethSrc = String.valueOf(instance.value(7));
        if (!ethDest.startsWith("01-0c-cd-01") && ethSrc.startsWith("01-0c-cd-01")) {
            return true; // it is an attack
        }
        return false;
    }
}
