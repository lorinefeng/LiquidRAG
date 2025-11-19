import { promises as fs } from 'fs'
import path from 'path'

const projectRoot = process.cwd()
const srcDir = path.resolve('e:/AI/LiquidRAG/head_portrait')
const publicDir = path.join(projectRoot, 'public')
const dstDir = path.join(publicDir, 'head_portrait')
const jsonPath = path.join(publicDir, 'images.json')

async function ensureDir(p) {
  try { await fs.mkdir(p, { recursive: true }) } catch {}
}

async function listImages(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true })
  const files = entries
    .filter(d => d.isFile())
    .map(d => d.name)
    .filter(n => /\.(jpe?g|png)$/i.test(n))
  return files
}

async function copyImages(files) {
  await ensureDir(dstDir)
  for (const f of files) {
    const src = path.join(srcDir, f)
    const dst = path.join(dstDir, f)
    try {
      await fs.copyFile(src, dst)
    } catch {}
  }
}

async function writeJson(files) {
  const items = files.map(name => ({
    name,
    url: `/head_portrait/${name}`,
    ext: path.extname(name).toLowerCase()
  }))
  await fs.writeFile(jsonPath, JSON.stringify({ images: items }, null, 2), 'utf-8')
}

async function main() {
  const files = await listImages(srcDir)
  await copyImages(files)
  await writeJson(files)
  console.log(`Generated ${jsonPath} with ${files.length} images`)
}

main().catch(e => { console.error(e); process.exit(1) })