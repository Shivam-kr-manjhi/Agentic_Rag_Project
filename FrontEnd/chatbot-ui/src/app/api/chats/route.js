import fs from "fs";
import path from "path";

const CHAT_DIR = path.join(process.cwd(), "chats");

if (!fs.existsSync(CHAT_DIR)) {
  fs.mkdirSync(CHAT_DIR);
}

/* GET */
export async function GET() {
  const files = fs.readdirSync(CHAT_DIR);

  const chats = files.map((file) => {
    const data = fs.readFileSync(
      path.join(CHAT_DIR, file),
      "utf-8"
    );

    return JSON.parse(data);
  });

  return Response.json(chats);
}

/* POST (Create / Update) */
export async function POST(req) {
  const body = await req.json();

  // Use chat id as filename
  const fileName = `chat_${body.id}.json`;
  const filePath = path.join(CHAT_DIR, fileName);

  fs.writeFileSync(
    filePath,
    JSON.stringify(body, null, 2)
  );

  return Response.json({ success: true });
}
