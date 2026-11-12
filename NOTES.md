# Flesh, bone, tendon, ligament, skin

Filed 2026-11-11. These are the programs, not a retelling. Read the files.

Scale on every one of them: 1.70 m man. His left is -x. Forward is +z. Feet on the ground.

## Bone

`skeleton.py`. 206 adult bones. Fused sacrum, fused hip bones, one sternum, patellae in, ear bones in. Sutures do not move. Synovial joints move inside anatomical limits. No flesh.

`bones.json` is the same skeleton as a list the viewer can load. A bone is a name, a polyline, and joint limits. Skin does not invent a bend.

## Muscle, tendon, ligament

`muscles.py`. 650 skeletal muscles, left and right counted apart. Each one is a whole muscle, not fibers. A ligament leaves the bone and meets the belly. Tendon length is the span times 0.1, clamped between 6 mm and 40 mm. Thickness and width are the muscle, in metres. No skin in this file.

`baldur-body.json` is that musculature placed on the skeleton and written out. Asleep, skin off, height 1.70 m, 206 bones, 651 muscle rows.

## Skin

`shell_shape.py`. One skin grown off the 206-bone skeleton. Cell size 5 mm. It writes a shell. It does not simulate what is under it.

`epidermis.ts`. The same shell in the viewer. Joint range comes from the bone. The skin is a surface.

`clay.py`. One skin. Volumes only say where the clay bulges. Nothing is simulated underneath. It starts from `public/prefabs/baldur-base-skin.obj`, which is not in this folder.

`rest_pose.py` and `shape_to_ref.py`. Rebuild that skin in the bind pose and hang both arms as one piece.

`pose_arms.py`. Arm pose on the finished skin. Partial rotation keeps the shoulder attached.

`prefab_shell.py`. One surface per visible zero point. The eyes do not store the body. They ask where a direction hits it.

## What is not here

The later shoulder, ear, and forehead edits that were packed into `body.bin` and put on architect.myfamiliar.ai were made on a mesh in the session. That mesh was not saved back as a program. Do not treat the live site as this folder. This folder is the bone, the 650 muscles with their tendons and ligaments, and the skin programs above.
