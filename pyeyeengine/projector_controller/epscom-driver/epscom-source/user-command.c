//! common linux userspace dependencies
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

//! userspace driver dependency
//! all libusb prefixed functions are documented in libusb
#include <libusb-1.0/libusb.h>

//! ----------
//! | MACROS |
//! ----------

//! DEBUG MODE

/*!
    Defined at compile level
    See Makefile
    0 = FALSE
    1 = TRUE
    when EPSCOM_DEBUG == 1
    logs will be displayed at stsndsrt output e.g. linux terminal
    *READ README FILE*

    regarding EPSCOM_ARGS_NUM first arg will allways be the file names
    that is just how C works
*/
#ifndef EPSCOM_DEBUG
#define EPSCOM_DEBUG 0
#endif

#if EPSCOM_DEBUG == 1

//! if debug is true then set EPSCOM logger
#define EPSCOM_DEBUG_LOG printf
#define EPSCOM_ARGS_NUM 3
#define EPSCOM_DEBUG_SLEEP(t) sleep(t)
#else

//! else get rid of all EPSCOM loggers in code segment
#define EPSCOM_DEBUG_LOG(...)
#define EPSCOM_ARGS_NUM 2
#endif

// #define LOG printf

//! EPSCOM (epson 1780W projector specific) Device details
#define EPSCOM_VENDOR_ID 0x04b8
#define EPSCOM_PRODUCT_ID 0x0514

//! in order to increase robustness
//! most operations will retry this amount of times
//! before failing
#define RETRIES 3

//! EPSCOM Values necessary for offset read
//! *libusb_cpu_to_le16* see little endian vs big endian
#define EPSCOM_DEVICE_wINDEX libusb_cpu_to_le16(0x00)
#define EPSCOM_INPUT_READ libusb_cpu_to_le16(0x02)
#define EPSCOM_STATUS_wVALUE libusb_cpu_to_le16(0x00)
#define EPSCOM_READ_wVALUE EPSCOM_STATUS_wVALUE

//! status buffer length
//! specifies the length of the answer from device
#define EPSCOM_STATUS_BUF_LEN 0x02

//! EPSCOM command characters specified by epson usermanuel
#define EPSCOM_EOL "\r"
#define EPSCOM_SOL ":"
#define EPSCOM_REQUEST "?"

//! length of the answer from device
#define EPSCOM_BUF_LEN 31

//! Windows control device (winusb / wcid)
//! Linux doesn`t know this exist linux thinks that it is talking to air
#define EPSCOM_WINUSB_DEVICE libusb_cpu_to_le16(0xC0)

//! Macros for terminating opration
#define EPSCOM_TERMINATE(STR) \
    printf(STR);              \
    exit(-1);

//! For claiming interfaces
#define EPSCOM_OUTPUT_IFACE 0x02

//! Terns out that it is not the actual control device input
//! but nonetheless used as mutex like logic
//! EPSCOM_WINUSB_DEVICE macro comments
#define EPSCOM_INPUT_IFACE 0x00

//! Can`t claim interface error handling with bitwise logic
//! can be replaced into multiple error codes if one wants
//! to avoid using bitwise logic
#define EPSCOM_CLAIM_ERROR 0x04

//! EPSCOM error/debug strings
#define EPSCOM_OUTPUT_STR "OUTPUT"
#define EPSCOM_INPUT_STR "INPUT"

//! functions declerations
int claim_device(libusb_device_handle *);
int claim_interface(libusb_device_handle *, int);
int write_device(libusb_device_handle *, unsigned char *);
int read_device(libusb_device_handle *, unsigned char *);
int clear_device(libusb_device_handle *);

//! mutex like logic
//! if intarfaces are claimed by previous access
//! epscom will exit with error
int claim_device(libusb_device_handle *dev)
{

    EPSCOM_DEBUG_LOG("CLAIMING INTERFACES:\n");

    //! claim output
    if (claim_interface(dev, EPSCOM_OUTPUT_IFACE))
    {

        return EPSCOM_OUTPUT_IFACE | EPSCOM_CLAIM_ERROR; //! return OUTPUT error code
    }

    //! claim fake input see macro comment
    if (claim_interface(dev, EPSCOM_INPUT_IFACE))
    {

        return EPSCOM_INPUT_IFACE | EPSCOM_CLAIM_ERROR; //! return INPUT error code
    }

    return 0;
}

//! actual claiming logic
//! see claim_device function comments
int claim_interface(libusb_device_handle *dev, int iface)
{
    EPSCOM_DEBUG_LOG("CLAIMING %s:\n", iface == EPSCOM_OUTPUT_IFACE ? EPSCOM_OUTPUT_STR : EPSCOM_INPUT_STR);
    int i = RETRIES;
    int ret;

    do
    {
        //! claim interface
        ret = libusb_claim_interface(dev, iface);

        //! if write successful then exit trying loop
        if (!ret)
        {

            EPSCOM_DEBUG_LOG("SUCCESS: %s CLAIMED IN %d TRIES:\n", iface == EPSCOM_OUTPUT_IFACE ? EPSCOM_OUTPUT_STR : EPSCOM_INPUT_STR, RETRIES - i + 1);
            i = RETRIES;
            return ret;
        }

    //! if write fail then try again i times
    } while (i--);

    EPSCOM_DEBUG_LOG("FAIL: CAN NOT CLAIM %s:\n", iface == EPSCOM_OUTPUT_IFACE ? EPSCOM_OUTPUT_STR : EPSCOM_INPUT_STR);

    return ret;
}

//! send cmd to device
int write_device(libusb_device_handle *dev, unsigned char *cmd)
{

    int buf_len = 0;
    int ret = -1;
    int i = RETRIES;

    do
    {
        //! write to device
        ret = libusb_bulk_transfer(dev, LIBUSB_ENDPOINT_OUT | 1, cmd, strlen(cmd), &buf_len, 0);

        //! if write successful then exit trying loop
        if (!ret)
        {

            cmd[strlen(cmd) - 1] = 0;
            EPSCOM_DEBUG_LOG("SUCCESS: SENT COMMAND \"%s\" IN %d TRIES %d BYTES SENT\n\n", cmd, RETRIES - i + 1, buf_len);
            strcat(cmd, EPSCOM_EOL);
            break;
        }

        //! if write fail wait 0.1 seconds and try again i times
        usleep(100000);

    } while (i--);

    //! if write failed after all retrys then
    if (ret)
    {

        cmd[strlen(cmd) - 1] = 0;
        EPSCOM_DEBUG_LOG("FAIL: FAILED SENDING \"%s\" COMMAND:\n", cmd);
        strcat(cmd, EPSCOM_EOL);
    }

    return ret;
}

//! read from device
int read_device(libusb_device_handle *dev, unsigned char *buf)
{
    int i = 0;
    unsigned char wlen_buf[EPSCOM_STATUS_BUF_LEN] = {0};
    int ret;

    //! read from device the length of the answer
    ret = libusb_control_transfer(dev, EPSCOM_WINUSB_DEVICE, LIBUSB_REQUEST_CLEAR_FEATURE, EPSCOM_STATUS_wVALUE, EPSCOM_DEVICE_wINDEX, wlen_buf, libusb_cpu_to_le16(EPSCOM_STATUS_BUF_LEN), 0);

    EPSCOM_DEBUG_LOG("STATUS READ BYTES LENGTH: %d\nREPORTED INPUT READ BYTES LENGTH: %u\n\n", ret, wlen_buf[0]);

    //! get answer length from length buffer
    ret = buf[0];

    i = RETRIES;

    do
    {
        //! keeping things save via waiting
        sleep(1);

        //! get answer length from answer buffer
        ret = libusb_control_transfer(dev, EPSCOM_WINUSB_DEVICE, EPSCOM_INPUT_READ, EPSCOM_READ_wVALUE, EPSCOM_DEVICE_wINDEX, buf, ret, 0);
        EPSCOM_DEBUG_LOG("TEMP INPUT COUNT: %d\n", ret);

    //! continue trying i times because excepting answer
    } while (ret <= 0 && i--);

    EPSCOM_DEBUG_LOG("TRUE INPUT COUNT: %d\n\n", ret);

    return ret;
}

//! clear buffer workaround instead of usb serial buffer clear instruction
//! due to lack of time for ferther investigating the winusb protocol
//! this is a workaround (and it seems to work exceptionaly well,
//! better than the windows driver) that clears the buffer from old data
//! it read from the device buffer until the buffer clear itself
int clear_device(libusb_device_handle *dev)
{

    EPSCOM_DEBUG_LOG("CHECKING FOR OLD DATA FROM PREV SESSION:\n");
    unsigned char clear_buf[EPSCOM_BUF_LEN];
    int ret = 0;
    int i = RETRIES;

    //! get data length from answer buffer incase it is not empty
    ret = libusb_control_transfer(dev, EPSCOM_WINUSB_DEVICE, LIBUSB_REQUEST_CLEAR_FEATURE, EPSCOM_STATUS_wVALUE, EPSCOM_DEVICE_wINDEX, clear_buf, libusb_cpu_to_le16(EPSCOM_STATUS_BUF_LEN), 0);

    //! if buffer is clear (clear_buf[0] == 0) then return success
    if (!clear_buf[0])
    {
        return clear_buf[0]; //! allways returns 0 sorry for complicating the code
    }

    //! prety sure this does nothing but too afraid to get rid of this
    ret = clear_buf[0];

    do
    {
        sleep(1);
        ret = libusb_control_transfer(dev, EPSCOM_WINUSB_DEVICE, EPSCOM_INPUT_READ, EPSCOM_READ_wVALUE, EPSCOM_DEVICE_wINDEX, clear_buf, ret, 0);
        EPSCOM_DEBUG_LOG("TEMP CLEAR COUNT: %d\n", ret);
    
    //! keep reading from device buffer until it clears itself up to i times
    //! after i times it assumes the buffer is frozen and the device requires
    //! a reboot
    } while (ret > 0 && i--);

    EPSCOM_DEBUG_LOG("TRUE CLEAR COUNT: %d\n\n", ret);
    return ret;
}

//! main function
int main(int argc, char const *argv[])
{

    int ret;
    char *ch;

    //! device pointer
    libusb_device_handle *epson;

    unsigned char buffer[EPSCOM_BUF_LEN] = {0};

    EPSCOM_DEBUG_LOG("EPSCOM DEBUG MODE:\n\n");

    //! check valid number of arguments
    if (argc != EPSCOM_ARGS_NUM)
    {
        printf("ERROR: %d ARGS REQUIRED ABORTING:", EPSCOM_ARGS_NUM - 1);
        EPSCOM_TERMINATE("\n");
    }

    EPSCOM_DEBUG_LOG("\n\nINIT:\nCMD: %s\nSLEEP: %s\n\n", argv[1], argv[2]);

    //! buffer = argv[1]
    strcpy(buffer, argv[1]);

    //! buffer = buffer + '\r'
    strcat(buffer, EPSCOM_EOL);

    //! initiate libusb MUST call this function before calling other libusb functions
    libusb_init(NULL);

    //! get EPSCOM device
    epson = libusb_open_device_with_vid_pid(NULL, EPSCOM_VENDOR_ID, EPSCOM_PRODUCT_ID);

    if (!epson)
    {
        EPSCOM_TERMINATE("ERROR: NO DEVICE OR RESOURCE BUSY\n\n");
    }

    ret = claim_device(epson);

    if (ret)
    {
        if ((ret ^ EPSCOM_CLAIM_ERROR) == EPSCOM_OUTPUT_IFACE)
        {
            EPSCOM_TERMINATE("ERROR: BUSY\n");
        }
        EPSCOM_DEBUG_LOG("NOT TERMINATING ON INPUT CLAIMING ERROR:\n\n");
    }

    //! if device buffer frozen then terminate
    if (clear_device(epson) > 0)
    {
        EPSCOM_TERMINATE("ERROR: PERSISTENT OLD DATA IN BUFFER:\n");
    }
    EPSCOM_DEBUG_LOG("BUFFER CLEAR MOVING ON:\n\n");

    ret = write_device(epson, buffer);

    //! if not buffer.contains("?") then don`t stdout answer
    if(!strstr(buffer, EPSCOM_REQUEST)) {

        return ret;

    }

    sleep(1);

    read_device(epson, buffer);

    sleep(1);

    ch = strstr(buffer, EPSCOM_EOL);
    if (!ch)
    {
        ch = strstr(buffer, EPSCOM_SOL);
        if (!ch)
        {

            return -1;
        }
    }
    *ch = 0;

#if EPSCOM_DEBUG != 1
    puts(buffer);
#else
    EPSCOM_DEBUG_LOG("EPSCOM ANSWER \"%s\"\n", buffer);
    EPSCOM_DEBUG_SLEEP(atoi(argv[2]));
#endif

    //! release clai device
    libusb_close(epson);

    return 0;
}
