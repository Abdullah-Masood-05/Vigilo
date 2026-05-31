DeepScreen — test build
=======================

Thanks for trying this. It is an exam-proctoring app: it watches a webcam and
flags things a proctor would care about, like nobody being there, two people
being there, or a phone appearing.

It is early software. Some of it does not work yet, and that is what I am
hoping you can help me find out.

There is nothing to install first. Everything it needs ships inside the
installer.


STEP 1 — INSTALL AND RUN
------------------------

Run the installer, then launch "DeepScreen Viewer" from the Start menu.

Windows may warn that the app is from an unknown publisher — it is not signed,
because signing certificates cost money. Click "More info" then "Run anyway".

You should see yourself, with a green box around your face.


STEP 2 — CLICK "ENROL FACE" ONCE
--------------------------------

Look at the camera and click the "Enrol face" button on the right. Do this
before anything else.

That teaches it what you look like, so it can tell later whether it is still
you sitting there. The label under "violations" will change from
"not enrolled" to "enrolled". It can take about five seconds.


STEP 3 — THINGS TO TRY
----------------------

Do these one at a time, and give each one a few seconds to register. Things
appear in the "violation log" in the bottom-left corner.

  1. LEAVE THE FRAME
     Stand up and walk out of shot for about 5 seconds, then come back.
     Expect: "no_face" appears in the log after ~2.5 seconds. When you come
     back it stays in the list but greys out and shows how long you were gone.

  2. HOLD UP YOUR PHONE
     Hold your phone up next to your face for about 10 seconds, screen
     facing the camera. Then put it down.
     Expect: "prohibited_object handheld_device" appears. It may take a few
     seconds — it deliberately waits to be sure rather than reacting to one
     frame. It should clear a few seconds after you put the phone away.

  3. HAVE SOMEONE ELSE LEAN IN
     Get someone to put their head in shot next to yours for a few seconds.
     Expect: "multiple_faces" appears.
     (A printed photo or a face on a phone screen may also work, but not
     reliably — do not worry if it does not.)

  4. LOOK AWAY FROM THE SCREEN
     Turn your head well to one side, or look right off to the side, and
     hold it for 3-4 seconds.
     Expect: "head_turned_away" or "gaze_off_screen".

  5. LOOK DOWN AT YOUR LAP
     As if reading something on your knees. Hold for 3-4 seconds.
     Expect: "gaze_off_screen".

Then — and this is the most useful part — JUST USE IT NORMALLY for five or ten
minutes. Read something, type, scratch your nose, drink tea, whatever. Behave
completely honestly.


WHAT TO SEND BACK
-----------------

Two things:

  1. A SCREENSHOT OF THE VIOLATION LOG at the end of your session.
     (Press Windows key + Shift + S to snip part of the screen.)

  2. ONE OR TWO SENTENCES about anything that looked wrong — especially
     anything that fired while you were behaving normally.

THE SECOND ONE MATTERS MOST. I can already test that it catches me holding a
phone. What I cannot test is what it wrongly accuses somebody else of, on a
different face, a different camera, different lighting and a different room.
If it flagged you for sitting still and reading, that is the single most
useful thing you can tell me.


KNOWN NOT TO WORK
-----------------

  * BOOK DETECTION DOES NOT WORK in this build. Holding up a book will not be
    flagged. This is a known limitation of the model, not a bug in your
    install — please do not spend time on it. Phone detection is the one
    worth testing.

  * The app does not save anything to disk yet, so nothing is recorded
    anywhere and nothing is uploaded. Closing it discards the session. This
    is why I need the screenshot.

  * It is not signed, so Windows will warn about it.


PRIVACY
-------

Nothing leaves your machine. There is no network connection, no upload, no
telemetry, and no file is written. The face enrolment is held in memory only
and is gone the moment you close the app. The only thing I ever see is
whatever you choose to send me.
