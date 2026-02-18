import { writeFile, mkdir, readdir } from "fs/promises";
import { join } from "path";
import { NextResponse } from "next/server";

const UPLOAD_DIR = join(process.cwd(), "uploads");

/* GET — return list of already uploaded filenames */
export async function GET() {
  try {
    await mkdir(UPLOAD_DIR, { recursive: true });
    const files = await readdir(UPLOAD_DIR);
    return NextResponse.json({ files });
  } catch {
    return NextResponse.json({ files: [] });
  }
}

/* POST — save uploaded files to /uploads */
export async function POST(req) {
  const formData = await req.formData();
  const files = formData.getAll("files");

  if (!files || files.length === 0) {
    return NextResponse.json({ error: "No files received" }, { status: 400 });
  }

  await mkdir(UPLOAD_DIR, { recursive: true });

  const saved = [];

  for (const file of files) {
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    const filePath = join(UPLOAD_DIR, file.name);
    await writeFile(filePath, buffer);
    saved.push(file.name);
  }

  return NextResponse.json({ uploaded: saved });
}